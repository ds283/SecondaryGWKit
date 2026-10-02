# Prompt 04 — record a `SIGTERM` as killed

**Campaign:** [`README.md`](README.md) · **Board items:** **K1**–**K3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 00 landed and reviewed. Independent of 01–03. Not dispatched while 05's change to
`quadsource_atol_sweep.py` is uncommitted (both edit `run_build`). `git status` clean.
`python -m RunRegistry list` shows nothing `running`, and `pgrep -lf 'gcs_server|raylet|ray::'` is
empty: this prompt runs real Ray, and must be able to tell its own processes from anyone else's.
**Closes:** `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`, on the
`datastore-integrity` board's §4, with its index row deleted.
**Opens:** whatever it finds that this prompt does not fix.
**Recommended model:** **Opus**. A signal handler on a process that Ray also handles, verified on
real Ray.

**Read first:**

1. [`README.md`](README.md): §0 item 4, §1, §4 (the one function this prompt ships), §5.
2. [`docs/a3-v2-readiness-audit.md`](../../docs/a3-v2-readiness-audit.md): **S1**–**S4**, and the
   closing section's *04*. These are measured; this prompt rests on them and does not re-measure
   them. Note which two points stay inferred.
3. The board entry this prompt closes, on `prompts/datastore-integrity/IMPLEMENTATION_STATE.md`
   §3.3, with its **Widened** line.
4. `docs/a3-v2-readiness/sigterm_probe.py` and its three outputs: the process bookkeeping this
   prompt's probe reuses.
5. `docs/gktk-remedial/scoped_pipeline_run.py`: `terminal_state`, `register`, `main`.
   `docs/handover/quadsource_atol_sweep.py`: `_load_scoped_driver`, `run_build`. `RunRegistry`:
   `begin` (its `root=`), `Run.finish`.
6. `RunRegistry/tests/test_pipeline_adoption.py`: what is already pinned about `terminal_state`.

---

## 1. What is wanted

Both drivers install a `terminal` handler **before** `ray.init`. `ray.init` replaces it at once with
its own, which is `sys.exit(signum)` in Python and absl's failure-signal handler in C. So once Ray is
up, `terminal` never runs (S1). A `SIGTERM` during `ray.wait` then reaches the driver as
`SystemExit(1)`, and `terminal_state(1)` records the run as `failed`/1. absl has also armed a 3 s
`SIGALRM`, which kills the process by `SIGABRT` 2.78–3.06 s after the signal unless
`signal.alarm(0)` cancels it (S2). The `finish(fingerprint=True)` that follows took 2.48 s on the A3
baseline, and will take longer on A3 v2 (S4). An exit that skips `ray.shutdown()` left a
`gcs_server` running in 7 of 12 runs (S3). Prompt 10's `resumed-b` recorded nothing at all.

The audit measured how to stop all of this. A Python handler installed **after** `ray.init`
replaces absl's at the C level. It runs on the main thread inside `ray.wait` and `ray.get` within
milliseconds, and no alarm is armed (S1). This prompt installs one.

---

## 2. What to change

**K1 — one handler, installed after `ray.init`, writing `killed` at once.**
- **One function, in `docs/gktk-remedial/scoped_pipeline_run.py`.** `run_build` already imports
  that module through `_load_scoped_driver`. Name it as you like; README §4 leaves the name to this
  prompt, and the log records it.
- **It is called from each driver's `_local_ray_init`**, straight after the real `ray.init`
  returns. The two drivers are `scoped_pipeline_run.main` and `quadsource_atol_sweep.run_build`.
  - `quadsource_atol_sweep.py` has a second `_local_ray_init` in `run_child` and a second
    `terminal` in `sweep`. Both belong to the sweep's own mode, which is not A3 v2's launch: leave
    them unchanged, and record whether they share the defect.
- **It handles `SIGTERM` and `SIGINT`, and is installed whether or not the run is registered.**
  Registration only decides whether a record is written; the shutdown in K2 matters either way.
- **On its first call the handler:**
  1. writes `run.finish("killed", exit_code=128 + signum)`, when there is a run: no fingerprint,
     at once;
  2. prints one line of its own to stderr, naming the signal and the time. Ray's
     `*** SIGTERM received` banner no longer prints once this handler is installed, and prompt 10
     identified an interruption by it;
  3. remembers the signal where the driver can read it;
  4. raises `SystemExit(128 + signum)`. It will reach the driver's frame as `SystemExit(1)`
     (S1). That is expected, and is why step 3 exists.
- **On any later call** — a second signal during the cleanup — it prints one line and returns,
  raising nothing. `SIGKILL` remains for a person who means it.
- **Before `ray.init`, keep a handler.** Each driver's existing `terminal` stays in place for the
  window before Ray exists, where record-and-die-by-default is right. Whether the new function also
  replaces that pre-`ray.init` handler is the *prompt's choice*, stated in the log.

**K2 — after the state: the fingerprint, then Ray, then the exit.** In each driver's
`except SystemExit` around the `exec` of `main.py`. By then `main.py`'s `with ShardedPool` has
unwound, so the store is closed.
1. **If the handler fired:**
   - call `signal.alarm(0)`. This guards a signal that reached Ray's handler before the driver's
     was installed (S2);
   - call `run.finish("killed", exit_code=128 + signum, fingerprint=True)`, best-effort, when there
     is a run. `finish` never raises for the fingerprint's sake;
   - call `ray.shutdown()`;
   - exit with status `128 + signum`, not 1.
2. **If it did not**, the existing path, unchanged: `terminal_state(exc.code)`, the fingerprint, the
   re-raise. Add `ray.shutdown()` before the re-raise only if it changes nothing a test pins.
   Record the choice in the log.
- **Factor the post-`exec` decision into one function** shared by both drivers, testable without
  Ray: given the run (or `None`), the exception, the remembered signal, and a shutdown callable.
  Each driver's body then shrinks to a call.

**K3 — `terminal_state`'s docstring made true.**
- The docstring says that a `SIGTERM` reaches the driver as `SystemExit(15)` after Ray's own
  shutdown has run. Audit S1 measured otherwise: it arrives as `SystemExit(1)`, with no shutdown,
  under an armed alarm.
- Rewrite the docstring to say what happens now, citing the audit.
- Keep the function's mapping. `RunRegistry/tests/test_pipeline_adoption.py` pins
  `terminal_state(15) == "killed"`, and README §5 rule 7 keeps that test. Its name records the old
  belief. Note that in the log; do not rename it.

---

## 3. Verification

**Unit tests**, under `RunRegistry/tests/` beside `test_pipeline_adoption.py`, with no Ray (README §5
rule 9):
1. **The handler, called directly with a stand-in run:**
   - its first call writes `killed`/143 with no fingerprint, remembers the signal, prints its line,
     and raises `SystemExit(143)`;
   - its second call writes nothing and raises nothing;
   - `SIGINT` gives 130.
2. **The post-`exec` function:**
   - signalled: it writes `killed`/143 with the fingerprint, calls `alarm(0)`, calls the shutdown
     callable once, and exits 143. It never writes `failed`, even though the exception's code is 1;
   - not signalled: every existing `terminal_state` outcome, exactly as today.
3. **Both drivers install it after `ray.init`.** Read both files with `ast`. Each `_local_ray_init`
   of the two drivers calls the function after the real `ray.init`. Each driver's `except
   SystemExit` calls the post-`exec` function.
4. The RunRegistry suite with every existing test unchanged; all six suites serially at the end.

**A probe on real Ray** (README §5 rule 9): a script under `docs/a3-v2-readiness/`, for example
`sigterm_handler_probe.py`, run by hand, with its output committed beside it as `.txt`.
- **Each variant is a child process** built as `sigterm_probe.py` builds its children, and it
  reuses that probe's process bookkeeping by import, not by copy. The child:
  - registers a `Run` with `RunRegistry.begin(..., root=<a temporary directory>)`. **Never
    `var/runs/`** (README §5 rule 10);
  - names as `results` a small store built in a temporary directory (`Datastore/tests/real_store_fixtures.py`);
  - goes through the real `ray.init` wrapper shape and the real installed function;
  - blocks in `ray.wait`, and is sent `SIGTERM` 2 s in;
  - runs the real post-`exec` function.
- **Variants, each run three times:**
  - the plain path;
  - a cleanup that sleeps 8 s before the fingerprint (no abort may come);
  - a second `SIGTERM` during the cleanup.
- **Each run reports:**
  - `status.json` read back: the state, the exit code, the fingerprint field;
  - when `killed` was first written, relative to the signal;
  - the child's exit status;
  - whether Ray's banner printed;
  - every `gcs_server`, `raylet` or `ray::` process the child started, still alive at +0.5, +2, +5
    and +10 s after it exits.
- **Acceptance is:** `killed`/143 in every run, written within a second of the signal; the
  fingerprint present; exit status 143; no abort; **and no `gcs_server` left in any run**. A
  survivor is a failure of K2, not something to kill and move past. The probe still kills its own
  survivors, `kill` then `kill -9`, and prints each pid. The output ends with the processes before
  and after, and the pids killed.

**Deliberate breakage** (README §5 rule 8). Each must make a test or the probe fail. Restore it
afterwards, and record each in the log as a diff, exactly as applied:
- (i) the handler is installed before `ray.init` instead of after (probe: Ray's banner returns, and
  the abort with it, or the state is `failed`);
- (ii) the handler does not write `killed` itself (unit test 1);
- (iii) the post-`exec` function uses `terminal_state(exc.code)` when signalled (unit test 2:
  `failed`);
- (iv) `ray.shutdown()` is not called (unit test 2; the probe should then show a `gcs_server` left
  in some run — record whether it did);
- (v) a second signal raises again (unit test 1).

---

## 4. Acceptance

1. K1–K3 as §2 states them, with the function's name and each *prompt's choice* in the log.
2. The unit tests, the probe and its committed output, and the deliberate-breakage record (i)–(v).
3. `[10-a-sigterm-…]` moved to the `datastore-integrity` board's §4, with a closing note naming this
   commit and the probe's output. Its index row deleted; the count and date corrected.
4. `black --check` clean. On this board: the §1 row, K1–K3, §3/§4, §5. All in one commit.

---

## 5. Stop conditions — stop and ask the user

- The probe leaves a `gcs_server` in any run with `ray.shutdown()` called.
- `killed` cannot be written before the process dies in some run.
- A Ray process the probe started survives `kill -9`.
- An existing test pins the pre-`ray.init` `terminal`, or the drivers' post-`exec` lines, so that
  K1 or K2 cannot land without changing it.
- Anything here needs `var/runs/` or a store under `var/`.

---

## 6. What this prompt does not do

- It does not run either driver's pipeline. A registered, `SIGTERM`ed and resumed build is 07's
  rehearsal (L1–L2).
- It does not touch `run_child`, `sweep`, `main.py`, `RunRegistry`, or the pool.
- It does not change `RunRegistry`'s `finish` or `fingerprint_store`, nor make the fingerprint
  faster.

---

## 7. The log and the board

`logs/04-record-a-sigterm-as-killed.md`, using README §5.1. In addition:
- the Ray version and the machine;
- the probe's wall time and the load average around it;
- the sequence of events after a `SIGTERM`, before and after this prompt, as two numbered lists.

`IMPLEMENTATION_STATE.md`: the §1 row for 04, K1–K3, §3/§4, §5.
