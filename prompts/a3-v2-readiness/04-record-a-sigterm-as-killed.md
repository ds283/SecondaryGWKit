# Prompt 04 — record a `SIGTERM` as killed

**Campaign:** [`README.md`](README.md) · **Board items:** **K1**–**K4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 00 landed and reviewed. Independent of 01–03. Not dispatched while 05's change to
`quadsource_atol_sweep.py` is uncommitted (both edit `run_build`). `git status` clean.
`python -m RunRegistry list` shows nothing `running`, and `pgrep -lf 'gcs_server|raylet|ray::'` is
empty: this prompt runs real Ray, and must be able to tell its own processes from anyone else's.
*Amended 2026-10-03 (user):* `pgrep` lists no Ray process **started from this repository**. Another
project's Ray cluster may run beside this prompt. The probe tells its own processes from it by
descent from its child, as `sigterm_probe.py` does, and never signals one it did not start. K4 is
what keeps a driver's `ray.init` off that cluster.
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
7. *Added 2026-10-03.* `RunRegistry/tests/test_store_fingerprint.py` `TestDrivers` and
   `_finish_calls`, which pin the drivers' `run.finish(` calls by `ast` (§3, unit test 5).

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
  1. remembers the signal where the driver can read it. *Amended 2026-10-03:* this was step 3. It
     comes first so that nothing the handler does afterwards can lose it;
  2. writes `run.finish("killed", exit_code=128 + signum)`, when there is a run: no fingerprint,
     at once;
  3. prints one line of its own to stderr, naming the signal and the time. Ray's
     `*** SIGTERM received` banner no longer prints once this handler is installed, and prompt 10
     identified an interruption by it. *Amended 2026-10-03:* a failure to print never stops step 4.
     A handler that prints while the main thread is inside a write to the same buffered stream gets
     CPython's `RuntimeError: reentrant call`;
  4. raises `SystemExit(128 + signum)`. It will reach the driver's frame as `SystemExit(1)`
     (S1). That is expected, and is why step 1 exists.
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
- *Amended 2026-10-03.* **The remembered signal decides, whatever the exception's type.** A signal
  that is handled outside a Ray call, or during an unwind, can reach the driver as something other
  than `SystemExit`: a `RuntimeError` from a guarded step, or a `KeyboardInterrupt`. Today's
  `except BaseException` branch would then write `failed` over `killed`. So the function takes any
  `BaseException`, and step 1 runs whenever the handler fired.
- *Amended 2026-10-03.* **Both paths, registered or not.** Today each driver `exec`s `main.py` with
  no `try` when `run is None`. K1's handler is installed either way, and the shutdown and the
  `128 + signum` exit matter either way, so the unregistered `exec` goes through the same function.
  With no signal and no run it re-raises exactly as now.
- *Amended 2026-10-03.* **A signal during the not-signalled path's own fingerprint.** In that case
  `Run.finish(..., fingerprint=True)` catches the handler's `SystemExit` in its `except
  BaseException`, writes the state it was given over `killed`, and re-raises. This covers `done`
  after a clean exit, and `failed` or `terminal_state(...)` after an error. What is recorded then
  is the *prompt's choice*, stated in the log and pinned by a unit test. `RunRegistry` is not
  changed (§6).

**K3 — `terminal_state`'s docstring made true.**
- The docstring says that a `SIGTERM` reaches the driver as `SystemExit(15)` after Ray's own
  shutdown has run. Audit S1 measured otherwise: it arrives as `SystemExit(1)`, with no shutdown,
  under an armed alarm.
- Rewrite the docstring to say what happens now, citing the audit.
- Keep the function's mapping. `RunRegistry/tests/test_pipeline_adoption.py` pins
  `terminal_state(15) == "killed"`, and README §5 rule 7 keeps that test. Its name records the old
  belief. Note that in the log; do not rename it.

**K4 — the real `ray.init` starts a local instance (added 2026-10-03, user).**
- **The defect, measured by the orchestrator on 2026-10-03 at `74270dd`.** Neither driver's
  `_local_ray_init` passes `address`. On a machine where `ray start` has left an address in
  `/tmp/ray/ray_current_cluster`, `ray.init` therefore resolves to that cluster:
  - Ray's own resolver gives `127.0.0.1:6379`, another project's cluster;
  - the bare call `ray.init(num_cpus=1, include_dashboard=False)` raises `ValueError: When
    connecting to an existing cluster, num_cpus and num_gpus must not be provided.`, before
    connecting.

  So today a launch on such a machine refuses to start. Without `num_cpus`, it would run on the
  other cluster. `address="local"` started a fresh instance beside it (GCS on port 65387), and
  `ray.shutdown()` left none of its processes.
- **The change.** `run_build`'s and `scoped_pipeline_run.main`'s `_local_ray_init` pass
  `address="local"` to the real `ray.init`. Nothing else in the call changes.
- `run_child`'s wrapper shares the defect. It is the sweep's own mode, and is left unchanged and
  recorded, as K1 does with it.
- 07's rehearsal and A3 v2's launch on the macstudio inherit this. The log says so in *State handed
  to the next prompt*.

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
   SystemExit` calls the post-`exec` function. *Amended 2026-10-03:* the call is reached on the
   unregistered path too, and the real `ray.init` call carries `address="local"` (K4).
4. The RunRegistry suite with every existing test unchanged **except the one named in 5**; all six
   suites serially at the end.
5. *Added 2026-10-03 (user), under README §5 rule 7.* `RunRegistry/tests/test_store_fingerprint.py`
   `TestDrivers.test_the_finish_calls` counts each driver's `run.finish(` calls by `ast`, split by
   whether they sit inside a function named `terminal`. At `74270dd` the counts are (3 outside,
   1 inside) for `scoped_pipeline_run.py` and (6, 2) for `quadsource_atol_sweep.py`. K2 moves
   `run_build`'s calls into a function of `scoped_pipeline_run.py`, so those counts cannot hold.
   - **The test may change in two respects only.** First, `_finish_calls` may recognise the new
     handler, by its name, as a handler, alongside `terminal`. Second, the expected counts may
     change to the new shape.
   - **Both assertions stay:** every call outside a handler passes `fingerprint=True`, and none
     inside one does.
   - **The log gives** the old and new counts, file by file, and why each changed.

**A probe on real Ray** (README §5 rule 9): a script under `docs/a3-v2-readiness/`, for example
`sigterm_handler_probe.py`, run by hand, with its output committed beside it as `.txt`.
- **Each variant is a child process** built as `sigterm_probe.py` builds its children, and it
  reuses that probe's process bookkeeping by import, not by copy. *Amended 2026-10-03:*
  `sigterm_probe.run_variant` hard-codes that probe's own `--child` command, so what is imported
  is the helpers (`ray_procs`, `all_procs`, `descendants`, `exists`, `fmt`, `short`). The new
  probe writes its own variant runner, and `sigterm_probe.py` is not edited. The child:
  - registers a `Run` with `RunRegistry.begin(..., root=<a temporary directory>)`. **Never
    `var/runs/`** (README §5 rule 10);
  - names as `results` a small store built in a temporary directory (`Datastore/tests/real_store_fixtures.py`);
  - goes through the real `ray.init` wrapper shape and the real installed function. *Amended
    2026-10-03:* that includes K4's `address="local"`;
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
- (v) a second signal raises again (unit test 1);
- (vi) *added 2026-10-03:* `address="local"` dropped from one driver's real `ray.init` (unit test 3);
- (vii) *added 2026-10-03:* the post-`exec` function consults the remembered signal only for a
  `SystemExit` (unit test 2, with a `KeyboardInterrupt` or `RuntimeError` after the handler fired:
  `failed`).

---

## 4. Acceptance

1. K1–K4 as §2 states them, with the function's name and each *prompt's choice* in the log.
2. The unit tests, the probe and its committed output, and the deliberate-breakage record (i)–(vii).
3. `[10-a-sigterm-…]` moved to the `datastore-integrity` board's §4, with a closing note naming this
   commit and the probe's output. Its index row deleted; the count and date corrected.
4. `black --check` clean. On this board: the §1 row, K1–K4, §3/§4, §5. All in one commit.

---

## 5. Stop conditions — stop and ask the user

- The probe leaves a `gcs_server` in any run with `ray.shutdown()` called.
- `killed` cannot be written before the process dies in some run.
- A Ray process the probe started survives `kill -9`.
- An existing test pins the pre-`ray.init` `terminal`, or the drivers' post-`exec` lines, so that
  K1 or K2 cannot land without changing it.
- Anything here needs `var/runs/` or a store under `var/`.
- *Added 2026-10-03.* The probe's child connects to a Ray cluster it did not start, or the probe
  would have to signal a process that is not its child's descendant.

---

## 6. What this prompt does not do

- It does not run either driver's pipeline. A registered, `SIGTERM`ed and resumed build is 07's
  rehearsal (L1–L2).
- It does not touch `run_child`, `sweep`, `main.py`, `RunRegistry`, or the pool.
- It does not change `RunRegistry`'s `finish` or `fingerprint_store`, nor make the fingerprint
  faster. Of the existing tests under `RunRegistry/tests/`, it changes only the one §3 item 5
  names.

---

## 7. The log and the board

`logs/04-record-a-sigterm-as-killed.md`, using README §5.1. In addition:
- the Ray version and the machine;
- the probe's wall time and the load average around it;
- the sequence of events after a `SIGTERM`, before and after this prompt, as two numbered lists.

`IMPLEMENTATION_STATE.md`: the §1 row for 04, K1–K4, §3/§4, §5.
