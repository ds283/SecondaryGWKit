# Log 01 — the run registry

**Prompt:** [`prompts/run-registry/01-the-run-registry.md`](../01-the-run-registry.md)
**Commit:** *(this commit)* — "Record long-running jobs on disk instead of in a conversation"
**Model:** Claude Opus 5
**Date:** 2026-09-22
**Result:** COMPLETE

## What shipped

**A new top-level package, `RunRegistry/`** — 534 lines across two modules, of which 333 are
statements and the rest docstrings, comments and blanks. No new dependency; standard library only.
No production compute path is in the diff, and `docs/handover/realistic_large_x.py` is untouched.

**`RunRegistry/__init__.py`** (447 lines) — the layout, the helper and the liveness rule.

- `begin(campaign, prompt, slug, purpose, …)` creates `var/runs/<run-id>/`, writes
  `manifest.json`, then writes `status.json` as `running`, and returns a `Run`. The manifest is
  complete on disk before `begin()` returns, so a crash in the first instant still leaves a record.
- `Run.record(unit, data)` appends one JSON object to `checkpoint.jsonl`, `flush`es, `fsync`s and
  beats. `Run.known()` reads it back keyed by unit. `Run.heartbeat(units_done, units_total, pid)`
  refreshes the heartbeat with whatever is now known. `Run.finish(state, exit_code)` sets a
  terminal state.
- `pid_alive(pid)` is `os.kill(pid, 0)`, with the reason not to pattern-match written into its
  docstring in one line, as prompt §1.5 requires. `liveness(status, stale_after)` returns
  `alive` / `stale` / `finished` / `unknown`; **alive** requires both `kill -0` and a heartbeat
  inside the window the caller supplies.
- `list_runs(root, stale_after)` returns every run directory, newest first, reporting rather than
  skipping one with no manifest, and ignoring loose files beside them.

**`RunRegistry/__main__.py`** (87 lines) — `python -m RunRegistry list [--root] [--stale-after]`.
One line per run: id, state, progress, age, purpose; a `!!` in the margin of any run whose state is
`running` but which is not, and a summary on stderr saying to go and look. It writes nothing.

**`RunRegistry/tests/test_run_registry.py`** (474 lines, **16 test methods**, 2.0 s) — §4's seven
requirements and the lister's graceful degradation. No Ray, no datastore, no network; every run
directory is in a `tempfile.TemporaryDirectory` and nothing under `var/` is read or written.

**The `CLAUDE.md` section**, "Long-running jobs — the run registry", stating §2's six rules —
discovery, liveness, launch, do not babysit, durability, provenance — in the voice of "Repository
mechanics", above it.

### The API, and what was rejected

Three calls and a loop: `begin()` → `known()` / `record()` → `finish()`. What was considered and
left out:

- **A context manager** (`with RunRegistry.begin(...) as run:`) that sets `failed` on an exception.
  Rejected: it only works when the process lives long enough to unwind, and the failure mode this
  campaign exists for — `SIGKILL`, a laptop sleeping, a quota interruption — never unwinds. A
  terminal state that is *sometimes* written is worse than one the lister infers from a stale
  heartbeat, because it looks reliable.
- **`open_run(path)`**, to attach to an existing run directory. Rejected because nothing needs it:
  a resume is a **new** run directory pointing at the **same** checkpoint path, which is why the
  manifest carries that path and why `checkpoint=` takes one.
- **A `units_total` derived from the checkpoint.** Rejected: a job that knows its total states it,
  and one that does not should show `?` rather than a number that looks measured.
- **Anything that writes to `var/runs/` from the lister** — a cache, an index, a `.DS_Store`-style
  marker. Rejected by §3's "no automatic cleanup": the lister opens files read-only, and that is
  the whole of its contract with the A3 evidence.

### Discard or refuse, on a script-hash mismatch

**Discard, with a notice on stderr** — `Run.known()`'s docstring says so. This is
`realistic_large_x.load_checkpoint`'s choice and the reason transfers: an edited script is then
self-healing, where refusing strands the operator on a file they must know to delete, and the
notice is what stops the discard being silent. Records are never blended across script versions;
an existing record is **never re-stamped** with the running hash, which is asserted by comparing
the checkpoint file byte for byte before and after a `known()` that discards
(`test_a_record_from_another_script_is_discarded_and_never_re_stamped`).

The hash is taken **once, in `begin()`**, and copied onto each record from the manifest. That is
not laziness: the source the interpreter is running is the source as it was at launch, so the
launch-time hash is the correct stamp for every unit the process completes, and re-reading the file
mid-run would stamp records with a version that never computed anything. It also costs no
subprocess per unit.

### Every manifest field, and the README §0 failure it traces to

Campaign README §5 rule 7: a field justified by nothing in §0 does not get built.

| Field | Failure it traces to |
|---|---|
| `run_id` | §0 item 2 — a run orphaned to `PPID 1` with nothing on disk naming it. |
| `created` | §0 items 2 and 3 — "whether it still matters"; two poll loops spun five hours on a script that had finished in the first minute. |
| `purpose` | §0 item 2 — nothing said what it was **for**; a stranger had to reconstruct it from `ps`, `lsof` and a stderr file. |
| `campaign`, `prompt` | §0 item 2 — nothing said **who owned it**. Also §1.1's "a stranger reading `ls var/runs/`". |
| `argv`, `cwd` | §0 items 2 and 5 — how the job was invoked is what a resume needs, and reconstructing it after the fact is what took `ps` and `lsof`. The A3 manifest carried this by hand, under `restart`. |
| `git_head`, `git_dirty` | §0.1 — the provenance stamp that paid for itself, and §0 item 5, where a resume ran against a tree nobody had recorded. |
| `script`, `script_sha256` | §0.1 — the stamp that proved in four seconds that the committed script was byte-identical to the one behind the published tables. |
| `unit`, `expected_units` | §0 items 1 and 3 — a three-hour measurement whose progress lived in memory, and a poll loop waiting for a file to reach seventeen lines. Progress must be a number the job declares, not a line count somebody guesses at. |
| `checkpoint` | §0 items 1 and 4 — results that lived in a list until the end, and a durable path that must be nameable and must not be a scratchpad. |

`status.json` carries exactly §1.3's fields: `state`, `units_done`, `units_total`, `heartbeat`,
`pid`, `exit_code`. Nothing else. There is no host, user, hostname, python version, environment
capture, priority, tag or parent-run field, because no failure in §0 would have been caught by one.

## Deviations from the prompt

1. **The self-match regression needs two processes, not one.** `IMPLEMENTATION CHOICE`, forced by
   measurement. Prompt §1.5 asks for "a liveness check performed from a process whose own command
   line contains the target's name". Written that way and run, the vacuity guard in the test fired:
   **macOS `pgrep` excludes itself *and all of its ancestors* unless `-a` is given** (`man pgrep`),
   so a lone poller cannot match itself there and the naive implementation would not have been
   fooled. Rather than reach for `-a` — which would be testing a flag nobody used — the test now
   reproduces the failure as it actually happened: a second poller waits on the same job with the
   target's name in its command line, and the checking process, *also* named after the target,
   matches **it**. That is §0 item 3 as recorded — nineteen loops, each matched by the other
   eighteen — and it is platform-independent. The property the prompt states is preserved and
   asserted: the checking process's own command line is verified to contain the target's name.
   This is a finding worth keeping: **the naive check is not self-matching but cohort-matching,
   and it gets more certainly unsatisfiable the more pollers there are.**
2. **`begin()` takes a `pid`, and `heartbeat()` takes one too.** `STRUCTURALLY REQUIRED` by §2
   rule 3. "Launch detached, manifest written first, pid recorded" cannot be done by a process that
   records only its own pid: the launcher writes the manifest, *then* spawns the detached child,
   and must record the child's. Default is `os.getpid()`, which is right for the other pattern —
   a script that registers itself.
3. **The package is 534 lines, not "a few dozen".** `IMPLEMENTATION CHOICE`, declared rather than
   hidden. 333 of those are statements (266 in `__init__.py`, 67 in the lister) and 201 are
   docstrings, comments and blanks — several of them required by the prompt itself (§1.5's reason
   in `pid_alive`, §1.4's stated discard policy in `known()`). I judged this **not** to meet §7's
   stop condition, which is "growing past a few hundred lines … the design has drifted into a
   framework": there is no scheduler, queue, daemon, UI, supervision, locking or deletion, the
   public surface is ten functions and one class with six methods, and every line serves one of
   §1.2–§1.6's six requirements. It is recorded here so that the judgement is the reader's to
   disagree with, and so that prompt 02 knows the budget is spent.
4. **`liveness()` returns four strings, not a boolean.** `IMPLEMENTATION CHOICE`. §1.6 needs to
   distinguish "running and stale" from "finished" to shout about the first, and a boolean would
   have to be read alongside `state` at every call site, which is the kind of two-field invariant
   that goes wrong. `alive` / `stale` / `finished` / `unknown`.

Nothing was classified `UNINTENDED DRIFT`.

## Verification performed

### 1. The self-match regression bites — the check that mattered most

A copy of the package was made in a scratch directory and its `liveness()` replaced with the naive
implementation: `pgrep -f <the run's name>`, alive iff anything matches (the naive copy also
records `run_id` in `status.json`, since that is the name such an implementation must look for).
The **same test module**, copied unmodified, was then run against each.

Against the naive implementation:

```
$ PYTHONPATH=. python -m unittest RunRegistry.tests.test_run_registry.TestSelfMatch -v
test_a_poller_named_after_the_target_does_not_report_it_alive ... FAIL
======================================================================
FAIL: test_a_poller_named_after_the_target_does_not_report_it_alive
AssertionError: 'alive' != 'stale'
- alive
+ stale
 : the registry reported a dead run as alive from a process named after it, with
   another poller for the same job alive beside it
Ran 1 test in 0.263s
FAILED (failures=1)
```

Against the shipped implementation, in the repository:

```
$ PYTHONPATH=. ./venv/bin/python -m unittest RunRegistry.tests.test_run_registry.TestSelfMatch -v
test_a_poller_named_after_the_target_does_not_report_it_alive ... ok
Ran 1 test in 0.273s
OK
```

The whole suite against the naive copy is `Ran 16 tests … FAILED (failures=3)`. Two of the three
are the naive `liveness`: the regression above, and
`TestLiveness.test_a_live_process_with_an_old_heartbeat_is_stale`, which fails because a
command-line match cannot see a heartbeat at all. The third,
`test_manifest_carries_the_provenance_triple`, is an artefact of *where* the copy lives rather than
of the patch — `git_provenance()` runs `git` in the package's own repository root, and the scratch
copy is not in a git tree, so `git_head` is `"unknown"`. It is recorded here so that the count is
not over-claimed.

**The test cannot pass vacuously.** Before asserting the verdict it asserts that the trap is armed:
that the checking process's own command line really does contain the target's name (read back with
`ps -ww -p <pid> -o command=`), that `pgrep -f <target>` really does return a live pid, and that
the pid it returns is the other poller's and **not** the dead job's. Both of the first two
assertions fired during development and stopped a test that would have proved nothing — once when
the token was absent from the sibling's argv, and once on the macOS ancestor-exclusion rule of
deviation 1 above.

### 2. The suites

| Suite | Baseline at `4741b86` | After |
|---|---|---|
| `ComputeTargets` | Ran 552, OK | Ran 552, OK |
| `CosmologyModels` | Ran 39, OK | Ran 39, OK |
| `LiouvilleGreen` | Ran 148, OK (skipped=1) | Ran 148, OK (skipped=1) |
| `AdaptiveLevin` | Ran 32, OK | Ran 32, OK |
| `Datastore` | Ran 10, OK | Ran 10, OK |
| `RunRegistry` | — (did not exist) | **Ran 16, OK** |

`THREE_BESSEL_DIAGNOSTIC_PLOTS` was not set. `black --check RunRegistry/` is clean.

### 3. The seven tests of §4, and the two beyond them

| §4 | Test | What it pins |
|---|---|---|
| 1 | `TestSelfMatch.test_a_poller_named_after_the_target_does_not_report_it_alive` | Above. |
| 2 | `TestManifest.test_manifest_is_on_disk_before_the_work_starts` | A subprocess calls `begin()` and then `os._exit(1)` — no unwinding, no `finally`. The manifest is on disk, parses, and carries the purpose and argv; `status.json` says `running`. |
| 2 | `TestManifest.test_manifest_is_not_mutated_afterwards` | The manifest's bytes are identical after two `record()`s, a `heartbeat()` that moves `units_total`, and a `finish()`. Plus `test_a_second_begin_will_not_overwrite_a_manifest`: a second `begin()` at the same instant raises `FileExistsError` rather than rewriting one. |
| 3 | `TestCheckpoint.test_round_trip_and_resume_skips_what_is_known` | Two units recorded, read back with their provenance, `units_done == 2`, and the resume computes only the third. |
| 4 | `TestCheckpoint.test_a_record_from_another_script_is_discarded_and_never_re_stamped` | The foreign record is not returned, the notice names it, and the file is **byte-identical** before and after, foreign stamp intact. `test_a_record_shadowed_by_a_foreign_one_is_dropped_not_blended` covers the case where the foreign record shadows a good one for the same unit: the unit is dropped, not blended. |
| 5 | `TestCheckpoint.test_a_torn_final_line_is_tolerated` | A truncated `{"unit": "c", "data": {"val` appended after two good records; both survive, the torn one is ignored. |
| 6 | `TestStatus.test_status_is_replaced_atomically_from_the_same_directory` | `os.replace` is called exactly once per write, the source ends in `.tmp` and is in the **destination's own directory**, and no `.tmp` is left behind. `test_a_reader_never_sees_a_half_written_status` runs a reader thread against 400 status writes: every read parsed, and the test refuses to pass if the reader got fewer than 50 of them. |
| 7 | `TestLiveness.test_a_run_whose_process_is_gone_is_stale_not_alive` | The heartbeat is deliberately **fresh** and the pid is a reaped child's, so only `kill -0` can tell — the converse of `test_a_live_process_with_an_old_heartbeat_is_stale`, where the pid is this very process and only the window can tell. |
| — | `TestLister.test_a_directory_with_no_manifest_is_reported_not_skipped` | Acceptance §6.2 as a test: a manifest-less directory is listed with `has_manifest` false and liveness `unknown`, and a loose file beside the run directories is ignored. |
| — | `TestLister.test_the_command_line_prints_every_run_and_shouts_about_stale_ones` | `python -m RunRegistry list` exits 0, prints the manifest-less directory, and puts the stale warning on stderr. |

### 4. `python -m RunRegistry list` against the real `var/runs/`

Read-only, and it degrades on the two A3 pilot directories, which have no manifest, and ignores
`realistic_large_x_cells.jsonl`, `run.out`, `run.pid`, `run.progress` and `.DS_Store`:

```
$ PYTHONPATH=. ./venv/bin/python -m RunRegistry list
   RUN              STATE      PROGRESS      AGE  PURPOSE
   a3-pilot-resume  unknown           -  17h 30m  (no manifest — predates the registry, or was not begun through it)
   a3-pilot         unknown           -  17h 30m  (no manifest — predates the registry, or was not begun through it)

2 run(s) under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: 2 unknown.
```

The mixed case — registered runs alive, stale and finished beside a pre-registry directory — was
exercised against a scratch root rather than against `var/runs/`, so that this prompt writes
nothing there:

```
   RUN                                         STATE      PROGRESS      AGE  PURPOSE
   a3-pilot                                    unknown           -       1s  (no manifest — predates the registry, or was not begun through it)
   handover-A2-finished-20260922T024155        done          60/60       1s  a job that finished
!! handover-A3-crashed-20260922T024155         running       12/13       1s  a job that says it is running and is not
   run-registry-01-demo-alive-20260922T024155  running        7/60       1s  a job that is running and beating

4 run(s) under …/demo: 1 alive, 1 finished, 1 stale, 1 unknown.
!! 1 run(s) say they are running and are not: the pid does not answer kill -0, or the
   heartbeat is older than 900 s. Nothing here has killed or cleaned anything — go and look.
```

The alive row is the launcher pattern of rule 3: the registering process spawned a detached child
and passed its pid to `heartbeat(pid=…)`.

### 5. `var/` is untouched

Nothing in this prompt wrote to `var/`. All 25 files under it are unchanged, the newest mtime being
**2026-09-21T09:13:49**, a day before this work; `var/runs/realistic_large_x_cells.jsonl` is still
60 lines and 60,484 bytes. Aggregate of the sorted per-file SHA-1s:
`42af290bf3bbe2af7f5a6210ce5995f493afa2da`.

## Limits — what this does not do, honestly

- **Per-unit resolution, and nothing finer.** The heartbeat is written by `record()` and by an
  explicit `heartbeat()`. A job whose unit takes three hours will be reported **stale** while it is
  legitimately working, unless it beats inside the unit or the reader widens `--stale-after`. There
  is no thread and no timer doing it in the background, deliberately: that would be a daemon by
  another name, and a heartbeat a dead process's thread could still emit is worse than none.
  `realistic_large_x.py`'s slowest cell was ~12 h projected, which is exactly this case.
- **A unit of work must be nameable and JSON-serialisable.** The checkpoint is keyed by a string.
  A job with no natural unit — `main.py`, whose checkpoint is its datastore — gets a manifest and a
  heartbeat and no `checkpoint.jsonl`, which is §0.2's distinction and is the right answer for it.
- **What the static self-match test does not prove.** It proves that *this* implementation, called
  from a process named after its target, answers from `kill -0` and a heartbeat. It does **not**
  prove that no future caller will shell out to `pgrep` itself: the rule lives in `CLAUDE.md` and
  in a docstring, and neither is executable. It also does not cover a poll loop written in shell
  that never imports this package at all — which is precisely how the nineteen were written.
- **`kill -0` cannot see pid reuse.** A recycled pid answers, so `kill -0` alone can say "alive" of
  a run whose process is long gone. The heartbeat is what closes that: a recycled pid is not
  writing our `status.json`, so the conjunction is sound where either half alone is not. A run
  killed and its pid recycled *within the staleness window* is the residual hole, and it is
  narrow rather than closed.
- **A terminal state is only as good as the process that sets it.** `finish()` is a call; a
  `SIGKILL` never makes it. That is why `state == "running"` plus a stale heartbeat is reported
  loudly instead of being rewritten to `killed` — the registry does not infer deaths, it reports
  what it cannot reconcile.
- **Two concurrent runs of the same job are not prevented** (§3: no locks, no exclusivity). They
  get two directories and two manifests, which is at least visible.
- **A registry nobody reads is worse than none**, because it looks like coverage. Rule 1 —
  run the lister at the start of a session — is the load-bearing half of this prompt, and it is a
  convention with no mechanism behind it. Nothing in this commit can make anybody run it.

## Observations not acted on

1. **`var/runs/` holds four loose files that are somebody's run and are not identifiable as such.**
   `run.out` (23,616 bytes), `run.pid`, `run.progress` and `realistic_large_x_cells.jsonl` sit at
   the top level beside the two A3 pilot directories. The lister ignores them, correctly — it lists
   directories. Three of the four are exactly the artefacts the registry now gives a home to, and
   which of the runs they belong to can only be inferred. **Not acted on:** the prompt forbids this
   package from touching anything under `var/`, and moving them would destroy evidence and break
   `realistic_large_x.py`'s checkpoint path. Opened as
   `[01-var-runs-holds-unattributable-loose-files]` on the board.
2. **`var/runs/.DS_Store` and `var/.DS_Store` exist**, 6,148 and 8,196 bytes. Noise; the lister
   ignores them because they are files. Not worth an issue.
3. **macOS `pgrep` excludes its own ancestors** (deviation 1). Not an issue in the tree — it is a
   fact about the platform that makes the naive check *more* insidious, not less: a single poll
   loop appears to work, and the failure appears only once there are two. Recorded in the test's
   docstring and in the board's §2 **G1** row so that it is not rediscovered.

## State handed to prompt 02

- `RunRegistry` is importable from the repository root, tested, and used by nothing. Adoption is
  prompt 02's, entirely.
- **The hazard prompt 02 must not trip is untouched here.** `docs/handover/realistic_large_x.py`
  is not in this diff, and `var/runs/realistic_large_x_cells.jsonl` is still 60 lines at script
  hash `c1cd3598c23a`. That script gates reuse on a hash of its own source, so **any** edit to it
  discards all sixty cells behind `REALISTIC-LARGE-X.md`'s published tables. `Run.known()`'s
  discard policy is the same policy as that script's, which is deliberate: adopting the registry
  does not change the semantics of reuse, only where the file lives — and note that the file's
  *path* is a manifest field precisely so that a registered run can point at the existing
  `var/runs/realistic_large_x_cells.jsonl` rather than starting a fresh one inside a run directory.
- `main.py` needs registering, not re-plumbing (README §0.2): `begin(...)` with no `checkpoint=`,
  a `heartbeat()` per work item if that is cheap, and `finish()`. Its checkpoint is its datastore.
- `Run.stdout_path` and `Run.stderr_path` exist and are used by nothing in this commit. They are
  §1.1's layout, and are there for prompt 02's `nohup … > "$run/stdout.log" 2> "$run/stderr.log"`.
