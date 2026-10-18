# Orchestrator — prompt 02, adopt the hard-kill route

Read [`../README.md`](../README.md) first: §0, §1, §2.2 (02's charter), §4, §5 (rules 1–8, 10, 11,
15), §6.2 (C3, C4, C5) and §8. Then the board's §5.1, with its review of 01. The conventions are
[`prompt-01.md`](prompt-01.md) §0's, carried over; what differs is said here.

**You do not write code.** You may:
- run the suites, both interrupter dummy tests, and `python -m runregistrykit list`;
- read RunRegistryKit's repository with `git`, never changing it;
- run a probe or a mutation from the session scratchpad, against a scratch copy, never committing
  one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`02-adopt-the-hard-kill-route.md`](../02-adopt-the-hard-kill-route.md)
**Board items:** A1, A2, A3 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 01 landed (`0a27b0c`) and reviewed (`c288eb8`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (outside `prompts/`
  and `docs/`).
- Nothing is `running` or stale in the registry; the rehearsal lock is free; `ray-processes` lists
  nothing. 02 takes no lock and starts no Ray, but its children are real processes, and the
  review's `var/` check needs a quiet tree.
- The venv holds `runregistrykit` 0.2.0 at `0bae3a3c…`, not editable.
- `git worktree list` shows the main checkout alone.

## 0. What makes this prompt unusual

**Its test signals a real process.** It is the first test in this repository that starts a child
and sends it `SIGKILL`. A child that outlives its test is a stray process with a `running` status
in a deleted directory; a test that leaves one has failed even if it passed. Every child is reaped
by `wait()` in the test and again by a cleanup registered before the first read of its output, as
RunRegistryKit's `_reap` does.

**Its restoration assertion is easy to make vacuous.** Prompt §2.3 test 1 asserts that `register`
took over neither stream nor handler before `begin` refused. `test_pipeline_adoption`'s helper
replaces both streams with `StringWriter`s, calls `register`, and puts everything back in a
`finally`. An assertion made *after* that helper returns sees what the `finally` restored, and
passes whatever `register` did: deliberate breakage 3 would then not be caught. **The observation
must be taken inside the helper, between the raise and the restoration**: catch `RuntimeError`,
record whether `sys.stdout`, `sys.stderr` and the two `signal.getsignal` values are the ones the
helper installed or found (by `is`), re-raise or return it, and restore in the `finally`. The
orchestrator's probe did exactly that (§0, the facts). Breakage 3 is how the review knows the
assertion is real.

**What moves on purpose:**
- `CLAUDE.md`, "Long-running jobs — the run registry" only:
  - rule 2 gains host and boot (prompt §2.1);
  - a **new rule 5**, the route after a hard kill; the old 5 (Durability) and 6 (Provenance) become
    6 and 7. Nothing outside `prompts/` names those two by number (`git grep` at `c288eb8`), so the
    renumbering breaks no reference;
  - the pointer sentence (`:93-95`) gains "Ending a run whose end was never recorded". Note that
    RunRegistryKit's "The rules" has six rules and **no** hard-kill rule: the pointer must say that
    the new rule's source is that section, not "The rules";
  - **one change the prompt does not name, which this note allows**: `:74-75` lists the command
    line as "`list`, and `store show | create | …`". `v0.2.0` added `end`, and the new rule names
    it. Add `end` to that parenthesis, and record it as a deviation, `IMPLEMENTATION CHOICE`,
    sanctioned by this note under README §6.2 C4 (01's pin made the list incomplete, and 01's
    review missed it). No other line of `CLAUDE.md` changes.
- `docs/datastore-integrity-verification/interrupt_when.py`: `--signal {TERM,KILL}`, default
  `TERM`; the docstring's first line, usage and one paragraph; the send and its log line.
- `docs/a3-v2-readiness/interrupter_build_dummy_test.sh`: a case D after case C, and its header
  comment's case list gains D.
- `RunDrivers/tests/test_hard_killed_predecessor.py`, new.
- `measure/out/02-roots.txt`, `02-dummy-build.txt`, `02-dummy-scoped.txt`.
- `logs/02-adopt-the-hard-kill-route.md`, new.
- The board: header, §1 row 02, A1–A3, `### 5.2 After prompt 02 (2026-10-18)`.
- No issue opens or closes unless the work finds one. The index stays at 100.

**The default must be byte-identical in its output.** With `--signal` absent the interrupter's log
lines are exactly those of today, the `watching` line included: the TERM send's line stays
`sent SIGTERM to pid … at stage … (…% >= …%); command …; exiting`. Build the name from the signal
(`signal.Signals(...).name` gives `SIGTERM` / `SIGKILL`), so the one f-string serves both. Do not
add the signal to the `watching` line; that would change cases A–C of both dummy tests and every
recorded output 03 compares with. `watch()` takes the signal as a keyword with default
`signal.SIGTERM`, so a caller that passes none is unchanged.

**The facts, checked by the orchestrator on 2026-10-18 at `c288eb8`** (`git diff --stat 0a27b0c
c288eb8 -- . ':!prompts' ':!docs'` is empty). Pass them on.
- **The tree and the registry.** Clean; one worktree. `list`: 45 runs, 43 finished, 2 unknown,
  nothing `running` or stale; "45 with no host recorded". Lock free; `ray-processes` none.
- **The package's route, probed with this repository's venv** (`runregistrykit` 0.2.0) from a
  scratch script that loads `scoped_pipeline_run` through `test_pipeline_adoption.load_driver()`:
  1. A child, `sys.executable -c CHILD` with `cwd` the repository root, calling
     `runregistrykit.begin(..., results=<never-created path>, root=<tmp runs root>,
     client_root=<repo root>)`, beating once and printing `run.path`, was `SIGKILL`ed; `wait()`
     returned `-9`; `status.json` byte-identical to before, `state` `running`, `host` `{"boot":
     "376461AF-…", "id": "8d2b247824882486"}`; `liveness(status, 900)` `stale`. The child imports
     the venv's installed `runregistrykit`; it needs no `sys.path` insertion, unlike the package's
     own `CHILD`.
  2. `register(args, ["--database", db], [1e5, 1e6], db, runs_root=root)` raised `RuntimeError` in
     0.02 s, with the message
     `Cannot begin run c-02-resume-<ts> with results "<db>": run example-01-hard-killed-<ts> is
     running (stale, pid <pid>) and names this store by its results. A stale run is ended by a
     person, with `python -m runregistrykit end RUN_ID --reason TEXT`, before another run is begun
     on its store; the registry does not decide that a run is dead. No run directory was created`
     — **no trailing full stop**. Both streams and both handlers were still the ones installed
     before the call, observed inside the helper; the root's listing was unchanged.
  3. `./venv/bin/python -m runregistrykit end <id> --reason probe --root <root>`, `cwd` the
     repository root: exit 0 in 0.2 s, stdout `>> end: <id> is now lost, verified: its pid <pid>
     does not answer kill -0 on this host`. The status: `state` `lost`, `exit_code` `null`,
     `ended` with `verified` `true`, `not_verified_because` `null`, `by`, `at`, `reason`,
     `git_head`, `host` (`id`, `name`, `boot`), and `evidence` with `host` `"this"`, `boot`
     `"this"`, `pid_answers` `false`, `state` `running`, `status_readable` `true`. Every other key
     kept.
     **`heartbeat` is written by the status writer**: RunRegistryKit's own `test_end.WRITTEN` is
     `{"state", "exit_code", "ended", "heartbeat"}`. The probe's value happened not to move (same
     second). The test's "every other key is kept" excludes those four, or it fails across a
     second boundary.
  4. A second `register` on the same path was then accepted, its run made in the root.
  5. A second `end` on the dead run: exit 1, stderr `!! Cannot end <id>: its status says it has
     already ended, as 'lost'; … Nothing was written`. (03 needs this; 02 may pin it too.)
  6. `end` on a **live** child: exit 1, stderr `!! Cannot end <id>: its pid <pid> answers kill -0,
     … --unverified does not override this. A process that was killed and not yet reaped by its
     parent (a zombie) answers kill -0 too, …. Nothing was written`; the status byte-identical.
  So none of the prompt's §4 package stops should fire. If one does, it is a stop, not a fix.
- **`register` and the client root.** `register` passes no `client_root`; `begin` walks up from
  `os.getcwd()`. The suites run from the repository root, so it finds `runregistrykit.toml` there;
  `runs_root=` overrides its `var/runs`. Use the **same string** for the database in the child and
  in `register`.
- **Both interrupter dummy tests at `c288eb8`, under `v0.2.0`**, give R2's outcomes
  (`docs/runregistry-extraction/rehearsal/interrupter_{build_,}dummy_test_08.txt`): build A
  `interrupter exit=0`, `child exit=143`; B `exit=2`, child alive; C `exit=2`, child alive.
  Scoped A `0`/`143`; B child `0`, interrupter `0`; C `2`, child alive; D `0`/`143`; E child `0`,
  interrupter `0`. Each log line in the same words, pids and paths aside. The dummy tests' statuses
  carry no `host`, so `v0.2.0` judges them by pid and heartbeat, as before.
- **Case D's child** is a `bash -c 'exec -a "python docs/handover/quadsource_atol_sweep.py --build
  --register dummy-build-d" sleep 120' &`, as case A's; `$!` is the `sleep`'s pid after the
  `exec`. `wait $CD` gives 137 under `SIGKILL`.
- **Black.** `interrupt_when.py` is black-clean at `c288eb8`; keep it so, and the new module.
- **The roots**, at `c288eb8`: §1's table.
- **The index:** 100 open; this campaign's one entry at §1.22.
- **Not touched, and staying:** `scoped_pipeline_run.py`, `quadsource_atol_sweep.py`, `main.py`,
  every reader, every existing test, `interrupter_dummy_test.sh`, `README.md`, `prompts/INDEX.md`,
  RunRegistryKit's checkout (`f6d4b80`, clean), the venv, and everything under `var/`.

**What the review exists to establish.**
- **(Q1) `CLAUDE.md`** says the prompt's §2.1 and this note's `end` in the command list, in the
  section's voice, and nothing RunRegistryKit's README at the tag contradicts.
- **(Q2) The interrupter.** `--signal KILL` sends `SIGKILL`, logs it, and changes nothing else;
  the default's output is byte-for-byte today's.
- **(Q3) The test** pins the refusal at SGK's own `register`, `end`'s acceptance and refusal, and
  the second registration; its children never outlive it; its restoration check is real.
- **(Q4) The mutations** of prompt §3.3 are each caught, and were applied as the log's diffs say.
- **(Q5) Nothing else moved**: no other file, no count but `RunDrivers`', nothing under `var/`.
- **(Q6) The records** in the campaign's form.

**Conventions,** prompt-01 §0's, with these changes:
- **The SHA.** "this commit" wherever 02's hash belongs; the orchestrator substitutes it.
- **Suites.** README §8's command, each root in the foreground, serially; the grep into
  `02-roots.txt` with the load average. A known flake is re-run alone and both runs recorded; it is
  not a stop.
- **Dummy tests.** Each into its `measure/out/02-dummy-*.txt`, whole, with a header line giving
  the command and `HEAD`. The work directory is in the agent's scratchpad, never `var/`.
- **Mutations.** Each applied to the working tree, run, recorded in the log as the diff exactly as
  applied with the failing test's name and assertion line, then reverted with `git checkout --
  <file>` or by hand; `git diff` against the final state shows none left. Breakage 5 is run on
  the build dummy test's case A.
- **`var/` is read, never written.** `list` only. No `begin()` with the default root, no `end`
  outside a test's temporary root, no `store` command.
- **Processes.** Before committing, `ps -o pid,ppid,command -U $USER | grep -E
  'hard-killed|sleep 120|sleep 30'` shows no child left by the agent's runs. Name any it finds.
- **Other sessions** may land commits on this branch. Check `git log` before measuring and before
  committing.

## 1. Before you dispatch

Done at `c288eb8` on 2026-10-18, before this note was committed; the results are §0's facts.
1. **The tree.** Branch `handover-remedial`; `git status --short` empty; one worktree.
2. **The registry.** Nothing `running` or stale.
3. **The lock and Ray.** Free; none.
4. **The probe** of §0, in the scratchpad's `orch02/`; its temporary directories hold only the
   probe's runs.
5. **Both dummy tests**, as §0.
6. **Baselines.** The seven roots, serially, at `c288eb8`:

   | Root | Ran | Result | Wall |
   |---|---|---|---|
   | `AdaptiveLevin/tests` | 32 | OK | 1 s |
   | `CosmologyModels/tests` | 39 | OK | 2 s |
   | `ObjectFactories/tests` | 279 | OK | 16 s |
   | `tools/tests` | 91 | OK | 13 s |
   | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 25 s |
   | `RunDrivers/tests` | 99 | OK | 4 s |
   | `ComputeTargets/tests` | 700 | OK | 137 s |

   Every count is 01's. Load average 8–24 on 10 cores, from other work; no known flake fired.

7. **The `var/` marker**, for check 5: immediately before dispatch, `touch` a marker file in the
   orchestrator's scratch directory and record its path.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README and the board, `HEAD`,
the baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a copy in
the session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's §6 list.
- **The restoration check is observed inside the helper** (§0), or breakage 3 is not caught.
- **`heartbeat` is excluded** from "every other key is kept" (§0, fact 3).
- **The interrupter's default output is unchanged**, line for line (§0).
- **`end` joins `CLAUDE.md`'s command list**, recorded as a sanctioned deviation (§0).
- **No child outlives a test**, a mutation run included.
- **The wall-clock flakes** are expected, re-run alone and recorded, not a stop.
- **Stop and ask** on any of the prompt's §4 conditions.

## 3. The review — seven checks

1. **Scope.** `git show --stat <02>`: `CLAUDE.md`, `interrupt_when.py`,
   `interrupter_build_dummy_test.sh`, the new test module, the three `measure/out/02-*` files, the
   log and the board; `docs/OPEN_ISSUES.md` only with a new §3 entry, with its count and date.
   Nothing else: no driver, no existing test, no `README.md`, `prompts/INDEX.md` or
   `orchestrator/`.
2. **Q1, `CLAUDE.md`.** `git diff <02>~1 <02> -- CLAUDE.md` touches the registry section alone
   (`end` in `:74-75`'s list aside). Read each changed rule against RunRegistryKit's README at the
   tag ("A run", "Ending a run whose end was never recorded", "The rules" 2): host and boot in
   rule 2; `foreign` not tested; earlier boot `stale` whatever the pid; no-host records judged as
   before; rule 5's steps, "on the host it ran on", no override for an answering pid,
   `--unverified` only for another host or an unsaying status, `lost`; `Run.finish` never by hand
   on another's run; the unreaped-process sentence and the `nohup sh -c` reaping. Rules 6 and 7 are
   the old 5 and 6 word for word. The pointer names the section and does not attribute rule 5 to
   "The rules".
3. **Q2, the interrupter.** `git diff` of `interrupt_when.py`: the option, the keyword default,
   the send, the log line, the docstring; every check before the send unchanged. Re-run both dummy
   tests here at `<02>`: every case's outcome and log wording as §0's, and case D `interrupter
   exit=0`, `child exit=137`, its log `sent SIGKILL to pid …`. `black --check` clean.
4. **Q3, the test.** Read the module whole against prompt §2.3 and §0 here: `load_driver` imported
   from `test_pipeline_adoption`; no `ray`; nothing under `var/`; each child reaped in the test and
   by a cleanup; the restoration observed inside the helper; the message checks (id, `stale`,
   `python -m runregistrykit end`, ends `No run directory was created`); the listing; `end`'s
   status checks with the four written keys excluded; the second `register`; the live-run refusal
   with the status byte-identical. Run the module here three times; time it; after each run, no
   stray child.
5. **Q4 and Q5, the mutations and nothing else.** Re-apply breakages 1, 3 and 4 here from the
   log's diffs, against a scratch copy where possible; each is caught as the log says; revert, and
   `git status --short` is clean. Re-run the seven roots at `<02>`: `RunDrivers` at 99 plus the
   new tests, every other root at §1's counts. `find var -newer <marker>` is empty, or names only
   writes by another process. RunRegistryKit's checkout unchanged at `f6d4b80`, clean.
6. **The measurements.** `02-roots.txt` has seven roots with counts and loads; the two
   `02-dummy-*.txt` are whole outputs with their command and `HEAD`, matching what the log claims.
7. **Q6, the records.** The board: header, §1 row 02 (✅, "this commit", the log), A1–A3 ✅,
   `### 5.2 After prompt 02 (2026-10-18)`. The log has every §5.1 heading, quotes each changed
   `CLAUDE.md` paragraph (prompt §3.4), classifies each deviation (the `end` list among them),
   records the five breakages as diffs. The commit message is in `CLAUDE.md`'s form, ending
   `Co-Authored-By: Claude Sonnet …`.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 02* paragraph in §5.2: what each
  check found, and anything handed on to 03. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log;
  - the board's line on orchestrator notes, listing this one as "used for 02";
  - `prompts/INDEX.md`'s row for this campaign: 2 of 4 landed;
  - README §2's 02 row status.
- Anything that is not residue is opened on the board's §3 and in the index.
- **Report to the user:** the rules as now written; the interrupter's option and its unchanged
  default; the test and what it pins; the mutations; the roots; what 03 inherits.
- **Then 03**, R3. Its note is written after this review, and it takes the rehearsal lock.
