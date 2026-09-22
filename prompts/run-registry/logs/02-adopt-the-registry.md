# Log 02 — adopt the registry, without falsifying anything

**Prompt:** [`prompts/run-registry/02-adopt-the-registry.md`](../02-adopt-the-registry.md)
**Commit:** *(this commit)* — "Register the pipeline driver without re-plumbing it"
**Model:** Claude Opus 5
**Date:** 2026-09-22
**Result:** COMPLETE

## What shipped

**`docs/gktk-remedial/scoped_pipeline_run.py` registers, behind an opt-in flag.** With
`--register <slug> --purpose "…"` the driver writes a `RunRegistry` manifest **before** the
pipeline starts, keeps `status.json` current while it runs, tees its stdout and stderr into the
run directory, and records a terminal state on exit. Without the flag the file behaves exactly as
it did: `register()` is never called, neither stream is wrapped, and the `exec` is the same one
statement it was — checked, not assumed (verification §1).

**`RunRegistry` gained three fields and one display rule**, each of them a field the hand-written
`var/datastores/handover-A3-baseline-lambdacdm.manifest.json` carried *by hand* and the package
did not have. That manifest is this prompt's requirements document: it is what somebody wrote
when they had to reconstruct a running job from `ps`, `lsof` and a stderr file.

| New | Where | The hand-written manifest's field |
|---|---|---|
| `scope` | `begin()`, manifest | `scope` + `note` — what the run covers, in a line a stranger reads. Derived from the job's own arguments, never typed. |
| `heartbeat_means` | `begin()`, manifest | (none — prompt §3 item 2 requires it) What a fresh heartbeat on *this* job does and does not assert. |
| `stage` | `heartbeat()`, `status.json`, lister | `stage_reached`: *"CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)"*, typed in by a human reading a stdout file. |
| `-` for no units | the lister only | A job that declares no total and has recorded no unit prints `-`, not `0/?`. |

**`RunRegistry/tests/test_pipeline_adoption.py`** — **14 new test methods**, ~3 s, no Ray, no
datastore, no network; every run directory is in a temporary one, and the two cases that read
`var/` skip when it is absent. Suite 16 → **30**. What they pin: the stage string against the
hand-written manifest's own `stage_reached`; that no denominator is invented; that the manifest
says what the heartbeat does not assert; that the stream passes every byte through and a dead tee
costs neither output nor the heartbeat; that a SIGTERM arriving as Ray's `SystemExit(15)` is
`killed` and not `failed`; and that a refused datastore leaves no run directory behind.

**`realistic_large_x.py` is not in the diff**, deliberately. §3 below.

### Which of prompt §3 item 1's two options, and why

**An opt-in flag, not a wrapper.** A wrapper would have to launch the driver as a child and then
either exit — leaving nothing to beat, so the run goes stale within fifteen minutes while it is
working — or stay and watch it, which is the supervision §5 forbids. The heartbeat has to come
from the process that is doing the work, and that process is the driver. So the flag.

**Why this is not a fourth change to `main.py`.** The driver's three changes are changes to
*main.py*: its `ray.init`, its model list, and its two grid literals. Registration adds nothing to
the execution namespace, alters no argument in `sys.argv`, substitutes no further text in the
source and reads no result. `main.py` is compiled from the same string and executed with the same
namespace under `--register` as without it; the flag is consumed by the driver's own parser before
the bare `--`, so it cannot reach `main.py`'s argv. What registration touches is the driver's own
process: a manifest before the exec, a stdout proxy that passes every write through unaltered, a
signal handler that records a death and then dies as the default would have, and a terminal state
after. It is two places in the file — a block before step (1), and a `try` around the `exec` —
not calls scattered through the substitution logic.

## What the heartbeat actually means

Prompt §3 item 2, and the thing this log is most careful about.

The work happens in Ray actors, so nothing the parent can write is evidence that an actor is
progressing. The heartbeat here is refreshed **when the pipeline writes a line to stdout**, at
most once every 30 s. That is worth more than a timer would be:

- `RayWorkPool` prints its progress line only *after* `ray.wait` has returned completed work
  (`RayTools/RayWorkPool.py:298` waits, `:548` prints), so a **fresh heartbeat does mean work
  completed recently** — strictly stronger than "the parent process is alive".
- A parent blocked for ever inside `ray.get` prints nothing, so it **goes stale**. That is the
  reading we want, and a timer thread would have reported it alive for ever.

**What it does not mean, said here, in the manifest as `heartbeat_means`, and in the driver's
docstring:** the converse does not hold. A stage that prints nothing for longer than the staleness
window — one long work item, a background-model build — is reported **stale while it is
legitimately working**. Stale means "go and look", not "dead", which is the lister's own wording.
It never means a Ray actor is progressing, and `kill -0` still cannot see pid reuse.

**No denominator was invented.** The pipeline reports progress per stage and per work queue and
never as one total, so `units_done`/`units_total` stay empty and the lister prints `-`. What it
reports instead is the `stage` string, in the form the hand-written manifest used. Fed the whole
of `var/runs/a3-pilot/run.out` — the real 10 h 33 m A3 stdout, read only — `StageTracker` ends at

    CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)

which is `stage_reached` in the hand-written manifest, character for character. The field is right
because somebody already needed it and had to produce it by hand.

## Deviations from the prompt

1. **`RunRegistry` was extended by three fields, where the prompt asked only for adoption.**
   `STRUCTURALLY REQUIRED` for two of them: prompt §3 item 2 says the heartbeat's meaning must be
   *in the manifest*, and prompt §3 item 3 asks for an honest `stage` where `begin()` and
   `heartbeat()` had nowhere to put one. `scope` is an `IMPLEMENTATION CHOICE`: prompt §1 says the
   datastore path and the grid criterion belong in the manifest, and `argv` alone is exact and
   unreadable. Campaign README §5 rule 7 is met — each traces to a failure in §0 **and** to a
   field the hand-written A3 manifest carried by hand, which is the strongest evidence a field
   will be read that this campaign has.
2. **The grid criterion is not a manifest field.** `IMPLEMENTATION CHOICE`, against the letter of
   prompt §1. The criterion is *code* — `main.source_grid_spacing_profile` at a given tree — so
   what identifies it is `git_head`, plus `--source-samples-log10z` in `argv` and `scope`. A
   hand-typed `"post qcd-background-audit prompt 15"` string would be exactly rule 7's field that
   goes stale: nothing can keep it true, and it would be wrong the first time somebody re-ran an
   old command on a new tree. Everything in `scope` is derived from the job's own arguments.
3. **The driver tees stdout *and stderr* into the run directory.** `IMPLEMENTATION CHOICE`.
   `CLAUDE.md` rule 3 says both streams go into the run directory, but the directory's name is not
   known until `begin()` has run, and here `begin()` runs *inside* the job — so a shell redirect
   cannot name it. The proxy that beats is already reading every line, so it writes a copy to
   `Run.stdout_path`, which prompt 01's log had reserved for exactly this. **stderr was added
   after the first demonstration attempt**, which crashed and left a run directory that could not
   say why: the traceback is on stderr. It is copied but does **not** beat and carries no stage —
   Ray's periodic warnings land there, and warnings are not progress. A tee that fails is dropped
   and never retried, and costs neither a line of output nor the heartbeat (asserted). The copies
   are closed only on the clean path, so the interpreter's traceback, printed after the frame
   unwinds, still reaches `stderr.log`.
4. **A `SIGTERM`/`SIGINT` handler records `killed`, and so does `terminal_state()`.**
   `IMPLEMENTATION CHOICE`, and deliberately not supervision: the handler writes the terminal
   state, restores the default handler and re-raises the signal at itself, so the process dies
   exactly as it would have. **The demonstration proved the handler is not the path that fires.**
   `ray.init` *replaces* it: `ray/_private/worker.py:1498` installs
   `def sigterm_handler(signum, frame): sys.exit(signum)`, so once Ray is up a `SIGTERM` arrives
   as `SystemExit(15)` — and the second demonstration recorded the run `failed`, exit code 15,
   which was **wrong about a run that was deliberately interrupted**. Ray's handler is the one
   that should win (it is why the A3 baseline's four shards passed `PRAGMA integrity_check` when
   it was stopped that way), so the driver reads the exit code instead: `terminal_state(15)` is
   `killed`. Our own handler still covers the window before `ray.init` and any run without Ray.
   Where no terminal state arrives at all — a `SIGKILL` — the registry reports the run **stale**,
   loudly, rather than inferring a death it did not see.
5. **The lister prints `-` rather than `0/?` for a job with no units.** `IMPLEMENTATION CHOICE`,
   display only. `0/?` reads as "no work has been done", which is more than the registry knows
   about a job that never counts units. This is a change to `RunRegistry/__main__.py`, which is
   prompt 01's file: what it does **not** touch is `status.json`'s own fields, `begin()`'s
   semantics or any test prompt 01 wrote — the first draft of this change was in `begin()`, it
   broke `test_a_reader_never_sees_a_half_written_status`, and it was withdrawn in favour of the
   display rule rather than adjusting a test of prompt 01's to accommodate it.

Nothing was classified `UNINTENDED DRIFT`.

## §3 — `realistic_large_x.py` is untouched, and that is the decision

`IMPLEMENTATION CHOICE`, taken deliberately, as prompt §2 directs.

That script gates checkpoint reuse on a SHA-256 of its own source. **One character discards all
sixty cells** behind every table in `docs/handover/REALISTIC-LARGE-X.md` and costs a three-hour
recomputation. Its value as a fixed artefact exceeds its value as a registry client: the hash is
what proved, in four seconds after the fact, that the committed script is byte-identical to the
one that produced the published numbers. Adopting the registry would buy it a manifest it does not
need — it has never been orphaned, its checkpoint path is in its own docstring, and its provenance
stamp is the pattern the registry copied — at the price of the one property nobody can buy back.

**Its records were not re-stamped, migrated or regenerated**, and no code in this commit can do
so. `Run.known()` discards a foreign record and recomputes its unit; prompt 01 asserts that byte
for byte. Nothing here read `var/runs/realistic_large_x_cells.jsonl` other than to count its lines
and hash its script hashes.

State before and after this commit, measured the same way both times:

| | before | after |
|---|---|---|
| `wc -l < var/runs/realistic_large_x_cells.jsonl` | **60** | **60** |
| script hashes in that checkpoint | **{`c1cd3598c23a`}**, exactly one | **{`c1cd3598c23a`}**, exactly one |
| `shasum -a 256 docs/handover/realistic_large_x.py` | `c1cd3598c23a5205…c2cd98a3` | `c1cd3598c23a5205…c2cd98a3` |

The script's own hash still begins `c1cd3598c23a`, matching the single hash in the checkpoint: the
committed script is still byte-identical to the one that produced all sixty cells.

`docs/source-remediation-verification/scoped_pipeline_run.py` is also untouched — it is broken by
`[13-scoped-run-driver-k-grid-literal]` and is another campaign's record.

## The worked example

Short, and not a framework. Two patterns, because README §0.2 distinguishes them.

**(a) A new measurement script — registry and checkpoint both.** Three calls and a loop:

```python
import RunRegistry

run = RunRegistry.begin(
    campaign="handover", prompt="04", slug="large-x-ladder",
    purpose="the realistic flavour at x_resp = 1e4 .. 1e8, one cell per rung",
    script=__file__, unit="cell", expected_units=len(PLAN),
    checkpoint=True,                  # inside the run directory; or a path, to share one
)
done = run.known()                    # records from another version of the script are discarded,
for item in PLAN:                     # with a notice, and never re-stamped
    if key(item) in done:
        continue
    run.record(key(item), compute(item))   # appends, flushes, fsyncs, beats
run.finish("done", exit_code=0)
```

Launch it detached, and then **leave it alone** (`CLAUDE.md` rule 4):

```bash
nohup env PYTHONPATH=. ./venv/bin/python -u docs/<campaign>/<script>.py \
    > /tmp/<slug>.out 2>&1 &
disown
PYTHONPATH=. ./venv/bin/python -m RunRegistry list     # one check that it started, then stop
```

The redirect is for diagnostics only, and it is allowed to be somewhere disposable: **the results
are in the checkpoint**, which `record()` has already `fsync`ed. A self-registering script cannot
name its own run directory in a shell redirect — the directory does not exist until `begin()`
returns — so if the output itself matters, copy it to `run.stdout_path` from inside the script,
which is what `scoped_pipeline_run.py` does.

**(b) A job whose checkpoint is not a JSON-Lines file** — a pipeline run, whose checkpoint is its
datastore. It registers; it is not re-plumbed. There is no `checkpoint.jsonl`, and `record()` and
`known()` are never called:

```python
run = RunRegistry.begin(
    campaign="handover", prompt="A3", slug="baseline-lambdacdm",
    purpose="the before-picture D1 is scored against",
    script=__file__,
    checkpoint=str(datastore_path),   # the durable thing the results live in
    scope=f"{len(k_sample)} log-spaced k over {k_min:g}-{k_max:g} /Mpc, …",   # derived
    heartbeat_means="…what a fresh beat on this job does and does not assert…",
)
...
run.heartbeat(stage="CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 remaining)")
...
run.finish("done", exit_code=0)
```

The rule the second pattern is written against: **say what the heartbeat means, and do not invent
a denominator.** If the honest answer is "the process produced output recently", write that in
`heartbeat_means` and let the lister call it stale when it stops.

## Verification performed

### 1. The demonstration — a short run, killed deliberately partway

**The demonstration of record is `var/runs/run-registry-02-demo-20260922T032842`.** Two k, two
cpus, one shard, a throwaway datastore at `var/runs/registry-02-demo-throwaway.sqlite` — under
`var/runs/`, never `var/datastores/`, never a scratchpad — launched detached with `nohup`/`disown`
and killed with `SIGTERM` at t = 150 s. The datastore and the nohup capture were deleted
afterwards; they were mine and they were throwaways. The run directory stays.

Two earlier attempts are also on disk, and the registry does not delete run directories:
`…T031844` (`failed`, exit 1) crashed on its own at 11 s because I asked for
`--source-samples-log10z 10`, which is too coarse for the curvature criterion — `build_z_sample`
refused it, correctly, and the registry recorded `failed` with the stage it had reached;
`…T032417` (`failed`, exit 15) is the SIGTERM that exposed deviation 4 — the kill worked, the
*label* was wrong, which is what `terminal_state()` now fixes.

**(i) t = 3 s — the manifest exists from the first second, and the lister shows the run.** Ray has
not finished starting; the record is already complete:

```
$ ls -l var/runs/run-registry-02-demo-20260922T032842
-rw-r--r--  1 ds283  staff  2255 Sep 22 03:28 manifest.json
-rw-r--r--  1 ds283  staff   227 Sep 22 03:28 status.json
-rw-r--r--  1 ds283  staff     0 Sep 22 03:28 stderr.log
-rw-r--r--  1 ds283  staff   262 Sep 22 03:28 stdout.log

$ PYTHONPATH=. ./venv/bin/python -m RunRegistry list
   RUN                                   STATE      PROGRESS      AGE  PURPOSE · STAGE
   run-registry-02-demo-20260922T032842  running           -       4s  prompt 02 demonstration: …  · list it with: PYTHONPATH=. ./venv/bin/python -m RunRegistry list
   run-registry-02-demo-20260922T032417  failed            -       4m  prompt 02 demonstration: …  · CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS
   run-registry-02-demo-20260922T031844  failed            -      10m  prompt 02 demonstration: …  · BUILDING ARRAY OF Z-VALUES AT WHICH TO SAMPLE
   a3-pilot-resume                       unknown           -  18h 18m  (no manifest — predates the registry, or was not begun through it)
   a3-pilot                              unknown           -  18h 18m  (no manifest — predates the registry, or was not begun through it)

5 run(s) under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: 1 alive, 2 finished, 2 unknown.
```

Both A3 pilot directories are still reported and not repaired, per prompt §3 item 4. The four
loose files beside them are ignored, and nothing in this run wrote to either.

**(ii) t = 150 s — `running`, alive, with a stage that is true.**

```
$ cat var/runs/run-registry-02-demo-20260922T032842/status.json
{
  "exit_code": null,
  "heartbeat": "2026-09-22T03:30:11+01:00",
  "pid": 50528,
  "stage": "CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS",
  "state": "running",
  "units_done": 0,
  "units_total": null
}

$ PYTHONPATH=. ./venv/bin/python -m RunRegistry list
   RUN                                   STATE      PROGRESS      AGE  PURPOSE · STAGE
   run-registry-02-demo-20260922T032842  running           -       3m  prompt 02 demonstration: …  · CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS
   …
5 run(s) under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: 1 alive, 2 finished, 2 unknown.
```

`PROGRESS` is `-`: no denominator was invented. `units_total` is `null` and the progress that
*does* exist is the stage string.

**(iii) after `kill -TERM` — `killed`, and not alive.**

```
$ kill -TERM 50528
$ kill -0 50528  ->  no such process

$ cat var/runs/run-registry-02-demo-20260922T032842/status.json
{
  "exit_code": 15,
  "heartbeat": "2026-09-22T03:31:13+01:00",
  "pid": 50528,
  "stage": "CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS",
  "state": "killed",
  "units_done": 0,
  "units_total": null
}

$ PYTHONPATH=. ./venv/bin/python -m RunRegistry list
   RUN                                   STATE      PROGRESS      AGE  PURPOSE · STAGE
   run-registry-02-demo-20260922T032842  killed            -       3m  prompt 02 demonstration: …  · CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS
   …
5 run(s) under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: 3 finished, 2 unknown.
```

The run is **`killed`**, its liveness is `finished`, and it is not alive. The stage records where
it was when it died — which is the sentence somebody had to reconstruct by hand for A3.

**What the run directory holds afterwards**, without anyone having had to arrange it: 73 lines of
`stdout.log` and 15 of `stderr.log`, including every stage the job passed through —

```
** scoped_pipeline_run: running models ['LambdaCDM']
>> RUNNING PIPELINE FOR MODEL LambdaCDM
** CALCULATE HORIZON EXIT TIMES FOR LambdaCDM SOURCE K-SAMPLE
** CALCULATE HORIZON EXIT TIMES FOR LambdaCDM RESPONSE K-SAMPLE
** BUILDING ARRAY OF Z-VALUES AT WHICH TO SAMPLE
** CALCULATING BACKGROUND LambdaCDM MODEL
** BUILDING BESSEL FUNCTION SPLINES FOR LambdaCDM
** CALCULATE NUMERICAL PART OF MATTER TRANSFER FUNCTIONS
** CALCULATE WKB PART OF MATTER TRANSFER FUNCTIONS
** CALCULATE QUADRATIC SOURCE TERMS
** CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS
** CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS
```

**The flag is opt-in, checked rather than assumed.** The same command without `--register` was run
first: it behaved as it always has, and **added nothing to `var/runs/`** beyond the throwaway
datastore it was given (`ls var/runs` identical before and after, once the throwaway was removed).
That run is also where the `--source-samples-log10z 10` traceback above was captured.

### 2. The suites

`PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .`, all six, against
the baseline taken at `b3cd0e3`. `THREE_BESSEL_DIAGNOSTIC_PLOTS` was not set.

| Suite | Baseline at `b3cd0e3` | After |
|---|---|---|
| `ComputeTargets` | Ran 552, OK | **Ran 552 in 143.4 s, OK** |
| `CosmologyModels` | Ran 39, OK | **Ran 39 in 0.7 s, OK** |
| `LiouvilleGreen` | Ran 148, OK (skipped=1) | **Ran 148 in 25.2 s, OK (skipped=1)** |
| `AdaptiveLevin` | Ran 32, OK | **Ran 32 in 0.05 s, OK** |
| `Datastore` | Ran 10, OK | **Ran 10 in 0.3 s, OK** |
| `RunRegistry` | Ran 16, OK | **Ran 30 in 2.6 s, OK** — **+14**, all in `test_pipeline_adoption.py` |

Every count is unchanged except `RunRegistry`, which rises by the fourteen this prompt wrote.
`test_tk_wkb_phase.TestCost.test_wall_time_per_object` — the known wall-clock flake — passed
within the full `ComputeTargets` run and was not re-run separately, there being nothing to
attribute. Nothing in this diff is on a production compute path: `ComputeTargets/`,
`CosmologyModels/`, `LiouvilleGreen/`, `AdaptiveLevin/`, `Datastore/`, `main.py`, `config/` and
`docs/spec/` are all absent from it.

### 3. `black --check`

Both files this prompt changed, and the new test module, were formatted with `black` and then
checked:

```
$ ./venv/bin/python -m black --check RunRegistry/ docs/gktk-remedial/scoped_pipeline_run.py
All done! ✨ 🍰 ✨
6 files would be left unchanged.
```

The remaining files in the diff are Markdown — this log, two campaign boards and
`docs/OPEN_ISSUES.md` — which `black` does not format.

### 4. The protected state, re-measured last of all

Taken again after every other step, with the same commands as the baseline:

```
$ wc -l < var/runs/realistic_large_x_cells.jsonl
      60
$ ./venv/bin/python -c "…script hashes in the checkpoint…"
['c1cd3598c23a']
$ shasum -a 256 docs/handover/realistic_large_x.py
c1cd3598c23a52051a0eb2695bf67977265383b5fa02977fb61d8251c2cd98a3
$ ls var/datastores/handover-A3-baseline-lambdacdm-shard000*.sqlite
87044096  …-shard0000.sqlite      87588864  …-shard0002.sqlite
87302144  …-shard0001.sqlite      86609920  …-shard0003.sqlite
$ du -sh var/datastores/backup-pre-resume-20260921T091011
333M
```

Identical to the baseline in every figure. `var/runs/a3-pilot/` and `a3-pilot-resume/` are
unchanged; the only thing this prompt did to them was **read** `a3-pilot/run.out`, to feed the
stage tracker and to attribute the loose files.

## Observations not acted on

1. **`[01-var-runs-holds-unattributable-loose-files]` can be attributed, and now is — narrowed,
   not closed, and nothing was moved.** Prompt 01's issue invited this prompt to record the
   attribution if it could be established from the file contents. It can, conclusively, and all
   four files are the *successful* `realistic_large_x.py` run of 2026-09-20 — not the 4 h 57 m
   attempt README §0 item 1 describes. `run.progress` is that script's **stderr**: its first line
   is the script's own checkpoint notice, naming `var/runs/realistic_large_x_cells.jsonl`, script
   hash `c1cd3598c23a` and HEAD `c414451b90d9`, and its last is
   `[ 10770 s] together x_resp=1e+08 realistic/open: 1515.5 s`. `run.out` is the same run's
   **stdout**: it opens with `<!-- generated by docs/handover/realistic_large_x.py -->` and closes
   with *"Cells: 60 computed or read from the checkpoint, 0 raised, 4 not reached. This
   invocation: 10772 s."* `run.pid` holds `21928` with mtime **14:09**, and the other two have
   mtime **17:08**; 17:08 − 10,772 s = **14:08:28**, so the pid file was written at that run's
   launch. **Not acted on:** campaign README §4 and prompt 01 §3 forbid moving, renaming or
   deleting any of them, and the attribution is now on the board rather than on disk. The issue
   stays open, narrowed.
2. **A registered pipeline run's `checkpoint` field names a SQLite file, which `Run.record()`
   would append JSON-Lines to.** Latent — nothing calls `record()` or `known()` on such a run, and
   the driver says so in a comment at the `begin()` call — but it is a loaded gun for the next
   script that copies the pattern and then adds units. Opened as
   `[02-a-datastore-checkpoint-path-would-be-appended-to-by-record]`. Not fixed here: the honest
   repair splits one manifest field into two, which is a change to prompt 01's data model that
   this prompt was not given.
3. **The stage tracker also matches the driver's own `**` banners.** At t = 4 s the demonstration's
   stage read *"list it with: PYTHONPATH=… -m RunRegistry list"* — the last line of the driver's
   own registration notice. It is accurate (that is genuinely the last thing printed) and it is
   replaced by a real stage within seconds, but a reader could mistake it for a pipeline stage.
   Not acted on: filtering the driver's own banners means the tracker would have to know which
   `**` lines are `main.py`'s, and a hard-coded exclusion list is exactly the thing that goes
   stale. Worth a line in whatever prompt next touches the tracker.
4. **`--source-samples-log10z 10` cannot build a source grid.** `build_z_sample` raises
   `ValueError: the spacing profile asks for a subdivision of 35 … above the
   SOURCE_GRID_MAX_REFINEMENT = 32 this construction will build; the largest asked for on the
   production envelope is 4`. This is the guard doing its job on a base lattice far coarser than
   production's 100, not a defect, and the message names the remedy. Recorded because it cost this
   prompt a demonstration run, and because it is a useful data point for anyone trying to make a
   *cheap* pipeline run: reduce `--k-count`, not the sample density.
5. **Three demonstration run directories are now in `var/runs/`**, all terminal, none marked `!!`.
   They are this prompt's evidence and the registry does not delete run directories. Removing them
   is the user's call, not this prompt's.

## State handed to the next prompt

- **The campaign is complete at 2 / 2.** `RunRegistry` is adopted by the one long-running entry
  point that needed it, and the pattern for the rest is in this log's worked example. Nothing in
  the tree schedules, supervises, restarts, locks or deletes anything.
- **The hazard is intact.** `docs/handover/realistic_large_x.py` is unchanged, its checkpoint is
  still 60 cells at the single hash `c1cd3598c23a`, and the script's own SHA-256 still begins
  `c1cd3598c23a` — so the property that proves the committed script produced the published tables
  survives this campaign. **No existing checkpoint record was re-stamped, migrated or
  regenerated**, here or in prompt 01.
- **The A3 baseline is untouched and still interrupted.** Its datastore, all four shards, the
  333 MB `backup-pre-resume-20260921T091011` and both pilot directories are byte-identical; this
  prompt only ever read the hand-written manifest and `a3-pilot/run.out`. Finishing that run is
  the user's, after `prompts/datastore-readback` prompt 01's fix — and when it is resumed, it can
  now be launched with `--register`, which is the whole point.
- **Two issues are open on this board and one is narrowed**, all three indexed at
  `docs/OPEN_ISSUES.md` §1.11. The one a future campaign is most likely to trip over is
  `[02-realistic-large-x-is-outside-the-registry]`, which exists so that the three-hour price of
  editing that script is known before anyone proposes it.
- **What is still a convention with nothing behind it** is `CLAUDE.md` rule 1: run the lister at
  the start of a session. Prompt 01 said so and it is still true — adoption by one driver does not
  make anybody look.
