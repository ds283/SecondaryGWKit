# Log — prompt 04: record a `SIGTERM` as killed

**Subject:** *Record a SIGTERM to a pipeline driver as killed* · **Model:** Claude Opus 5.5 ·
**Date:** 2026-10-03 · **Base:** `8690dc6`, clean · **Result:** landed, with one acceptance
criterion short. Both drivers start a fresh local Ray instance (`address="local"`) and then
install one handler for `SIGTERM` and `SIGINT`. A `SIGTERM` in `ray.wait` is recorded
`killed`/143 by the handler, then gains its fingerprint. Ray is shut down, and the process exits
143. On real Ray that held in 27 of 27 runs: the fingerprint was present, there was no abort and
no `gcs_server` was left. **The criterion "written within a second of the signal" failed in 4 of
the 27, at 1.003–1.006 s.** The cause is Ray's own check for signals inside a wait, which the
probe measured at the next whole second of the wait; load does not explain it (see *Verification
performed*, and `[04-a-sigterm-in-ray-wait-is-handled-up-to-a-second-late]`). No stop condition
of the prompt's §5 fired.
`[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]` is closed on the
`datastore-integrity` board. Four issues are opened. The commit is `79ab21a`.

The orchestrator's facts for this prompt, checked at `74270dd`, are cited below as "the
orchestrator's" where this prompt relies on them without re-deriving them.

**Ray and the machine.** Ray 2.43.0, Python 3.12.14, Darwin 25.5.0 arm64, MacBookPro18,1 (Apple
M1 Pro, 10 CPUs, 16 GiB), host `mps026088.phys.susx.ac.uk`. Another project's Ray cluster
(ChamPBH: `gcs_server` 62525 on `127.0.0.1:6379`, named by `/tmp/ray/ray_current_cluster`) ran
throughout. No process of it was signalled, connected to or stopped. Each probe run's own GCS was
on another port (for example `127.0.0.1:52707`). The ChamPBH processes are listed unchanged before
and after every probe invocation.

## What shipped

Line counts are `git diff --cached --numstat` of `79ab21a`.

- **`docs/gktk-remedial/scoped_pipeline_run.py`** (289 added, 33 deleted):
  - New, at module scope, with nothing imported from `ray`: `KILLED_SIGNALS`, `SignalLatch`
    (with `decide()`), `make_killed_handler` (whose handler is the nested `killed_handler`),
    **`install_killed_handler`**, **`finish_after_exec`**, and the private `_say`, `_signal_name`,
    `_now`, `_shut_down` and `_code_text`.
  - `terminal_state`: the docstring is rewritten (K3). The code is unchanged (`ast.unparse` of
    its body after the docstring equals `8690dc6`'s).
  - `main`: `_local_ray_init` passes `address="local"` to the real `ray.init` (K4), then calls
    `install_killed_handler(run, latch)`, and returns the context (K1). The `if run is None:
    exec; return` branch and the `except SystemExit`/`except BaseException` pair are replaced by
    one `try: exec … except BaseException as exc: finish_after_exec(run, exc, latch,
    ray.shutdown); raise`, then `finish_after_exec(run, None, latch, ray.shutdown)` and the
    existing `close_copy` loop (K2).
  - The module docstring: item 1 says `address="local"` and the handler, and a *SIGNALS*
    paragraph is added.
  - `register` is unchanged, `terminal` included (`ast.unparse` equal).
- **`docs/handover/quadsource_atol_sweep.py`** (16 added, 13 deleted), `run_build` only:
  - the same `_local_ray_init` change, through `scoped.SignalLatch` and
    `scoped.install_killed_handler`;
  - the same `try`/`finish_after_exec`/`raise` shape in place of the two `exec` branches;
  - a three-line comment.
  - `run_child`, `sweep` and every other function are unchanged (`ast.unparse` of each top-level
    function against `8690dc6`: only `run_build` differs).
- **`RunRegistry/tests/test_killed_handler.py`** (new, 578 lines, 26 tests), below.
- **`RunRegistry/tests/test_store_fingerprint.py`** (10 added, 4 deleted), under §3 item 5 only:
  - `_finish_calls` treats a function named `killed_handler` as a handler, beside `terminal`,
    through a module tuple `_HANDLERS` with a three-line comment. Its docstring says "signal
    handler" for "terminal handler";
  - `TestDrivers`' expected counts change. Both assertions are unchanged (see *The changed test*).
- **`docs/a3-v2-readiness/sigterm_handler_probe.py`** (new, 690 lines) and its output
  **`sigterm_handler_probe.txt`** (new, 2506 lines, three invocations), below.
- Not touched: any `RunRegistry/*.py`, `main.py`, the pool, `sigterm_probe.py` and every other
  probe, every other test, `var/`, `prompts/a3-v2-readiness/orchestrator/`.

### The functions, and what each does

- **`install_killed_handler(run, latch, stream=None)`** is the one function of README §4. It
  builds the handler with `make_killed_handler` and installs it for `SIGTERM` and `SIGINT`
  (`KILLED_SIGNALS`), and returns it. It is called with `run=None` on an unregistered run.
- **The handler, `killed_handler(signum, frame)`.**
  - **Its first call, while `latch.decided` is false:**
    1. sets `latch.signum = signum`;
    2. calls `run.finish("killed", exit_code=128 + signum)` when `run` is not `None`, inside
       `try/except Exception`, printing a failure;
    3. prints one line through `_say`, which never raises;
    4. raises `SystemExit(128 + signum)`.
  - **Any other call** (a later signal, or any signal once `latch.decided` is set) prints one
    `ignored` line and returns.
- **`SignalLatch.decide()`** sets `decided = True` if no signal has been remembered, and returns
  `signum`.
- **`finish_after_exec(run, exc, latch, shutdown, *, alarm=signal.alarm, stream=None)`.**
  - **First:** `signum = latch.decide()`. If the handler raises out of `decide()` (a signal between
    its test and its assignment), the `SystemExit` is caught and `latch.signum` used. Then
    `alarm(0)`, printing a line if it returned non-zero.
  - **Not signalled** (`signum is None`): `done`/0 for `exc is None`,
    `terminal_state(exc.code)`/`exc.code` for a `SystemExit`, `failed`/1 for anything else, each
    with `fingerprint=True`, only when there is a run. Then `shutdown()` (through `_shut_down`,
    which prints a failure and does not raise), and **return**. The driver re-raises with a bare
    `raise`.
  - **Signalled:** one line naming what reached the driver, then `run.finish("killed",
    exit_code=128 + signum, fingerprint=True)` in `try/except Exception`, then `shutdown()` in a
    `finally`, then `raise SystemExit(128 + signum)`.

### The prompt's choices

1. **The names.** `install_killed_handler` (README §4's one function), `finish_after_exec` (the
   post-`exec` function), `make_killed_handler` / `killed_handler` (the handler), `SignalLatch`
   (where the signal is remembered).
2. **The handler before `ray.init`.** Each driver's existing `terminal` stays, unchanged, for the
   window before Ray exists. The new function does not replace it. `register()` and `run_build`'s
   `if args.register:` block are unchanged.
3. **`ray.shutdown()` on the not-signalled path: added**, after the record and before the
   re-raise. No test pins its absence: `test_pipeline_adoption` passes untouched (15 tests). It
   changes no recorded state or exit status (`TestNotSignalled.test_every_outcome_is_recorded_as_it_was`).
   Every exit of either driver after `ray.init` now goes through `ray.shutdown()`, which audit S3
   found left no `gcs_server` in 10 of 10.
4. **A signal during the not-signalled path's own fingerprint** (amended K2). More generally, this
   covers any signal once the pipeline's own outcome has reached `finish_after_exec`. **The
   pipeline's outcome is recorded, with its fingerprint, and the exit is the pipeline's.** The
   handler sees `latch.decided`, prints `ignored, the pipeline had already ended; its own outcome
   is being recorded`, writes nothing and raises nothing. So `Run.finish` is never interrupted,
   and nothing is written over `killed`, because `killed` is never written.
   - *Why.* By then the pipeline has ended on its own: `done` says the store is complete, and
     `failed` is a real failure, which `killed` would hide. `Run.finish`'s own design for an
     interrupt while fingerprinting is the same: "the state is still written". Letting the
     handler raise there would leave `fingerprint_error: "interrupted: SystemExit"` beside the
     outcome. A failed run's traceback would also be lost, because a `SystemExit` raised inside
     the driver's `except` replaces it.
   - *Pinned by* `TestNotSignalled.test_a_signal_during_its_own_fingerprint_changes_nothing` (8
     outcomes, stand-in run), `test_on_a_real_run_the_outcome_keeps_its_fingerprint` (a real
     `Run`: `failed`/1, the fingerprint field written, no `interrupted`), and
     `TestTheHandler.test_once_the_outcome_is_decided_a_signal_is_only_noted`.
5. **The handler's line**, to `sys.stderr` as it is when the line is written (the
   `RegisteredStream` copy into `stderr.log` on a registered run): `** SIGTERM at 2026-10-03
   16:33:28 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit
   with status 143` (probe, `second-sigterm #1` of the third invocation). It never contains Ray's
   `*** SIGTERM received`, which the probe counts.

### The changed test (§3 item 5)

`TestDrivers.test_the_finish_calls` counts each driver's `run.finish(` calls by `ast`, as
`(outside a handler, inside one)`:

| Driver | At `8690dc6` | After | Why |
|---|---|---|---|
| `scoped_pipeline_run.py` | (3, 1) | (4, 2) | Outside: `main`'s three calls (`terminal_state`, `failed`, `done`) leave `main` and become `finish_after_exec`'s four (the same three, plus the signalled `killed`), each `fingerprint=True`. Inside: `register`'s `terminal`, plus `killed_handler`'s no-fingerprint `killed`. |
| `quadsource_atol_sweep.py` | (6, 2) | (3, 2) | Outside: `run_build`'s three calls move into `scoped_pipeline_run.finish_after_exec`, through `scoped.`; `sweep`'s three remain. Inside: `run_build`'s and `sweep`'s `terminal`, unchanged. |

Both assertions are unchanged: every call outside a handler passes `fingerprint=True`, and none
inside one does. Without the second `_HANDLERS` name the handler's call would count as outside,
with no fingerprint, and the first assertion would fail. Breakage (ii) shows the test seeing the
handler: deleting its `run.finish` gives `(4, 1) != (4, 2)`.

### The tests (`RunRegistry/tests/test_killed_handler.py`, 26)

No Ray, nothing under `var/`. A real `Run` is used only in a temporary run root. The handler is
called directly; `shutdown` and `alarm` are recorders.

- **Unit test 1 — the handler.** `TestTheHandler` (8):
  - `test_its_first_call_writes_killed_143_remembers_prints_and_raises`: `[("killed", 143,
    False)]`, `latch.signum == 15`, one line, `SystemExit(143)`;
  - `test_its_second_call_writes_nothing_and_raises_nothing` (a second `SIGTERM` and a `SIGINT`);
  - `test_sigint_gives_130`;
  - `test_unregistered_it_writes_nothing_and_still_raises`;
  - `test_a_print_that_fails_does_not_stop_the_raise` (the stream raises `RuntimeError:
    reentrant call …`);
  - `test_a_write_that_fails_does_not_stop_the_raise`;
  - `test_on_a_real_run_status_json_says_killed_143_and_holds_no_fingerprint`;
  - `test_once_the_outcome_is_decided_a_signal_is_only_noted`.

  `TestInstall` (2): `test_it_takes_sigterm_and_sigint`, and
  `test_installing_again_keeps_the_remembered_signal`, the orchestrator's re-install case.
- **Unit test 2 — the post-`exec` function.**
  - `TestSignalled` (8):
    - `test_it_is_killed_143_with_the_fingerprint_whatever_reaches_the_driver`, 6 sub-tests:
      `SystemExit(1)`, `SystemExit(143)`, `SystemExit(None)`, `KeyboardInterrupt`,
      `RuntimeError`, and no exception. Each gives `[("killed", 143, False), ("killed", 143,
      True)]` and no `failed`, `alarm(0)` once, shutdown once, and `SystemExit(143)`;
    - `test_sigint_is_killed_130`;
    - `test_unregistered_it_shuts_down_and_exits_143`;
    - `test_the_alarm_is_cancelled_before_the_fingerprint`: the order is `alarm(0)`, fingerprint,
      shutdown, and a 3 s alarm is reported;
    - `test_a_write_that_fails_still_shuts_down_and_exits`;
    - `test_a_shutdown_that_fails_still_exits_143`;
    - `test_a_second_signal_during_the_fingerprint_is_ignored`, on a real `Run`;
    - `test_a_signal_inside_decide_is_a_signal_that_came_first`.
  - `TestNotSignalled` (5):
    - `test_every_outcome_is_recorded_as_it_was`, 8 sub-tests. Each is checked against `today()`,
      a transcription of the drivers' code at `74270dd`: done/0, done/None, done/0, failed/1,
      killed/15, failed/'a message', failed/1 (`RuntimeError`) and failed/1 (`KeyboardInterrupt`).
      In each the function returns, shutdown and `alarm(0)` run once, and `decided` is set;
    - `test_unregistered_it_records_nothing_and_returns` (8);
    - `test_a_signal_during_its_own_fingerprint_changes_nothing` (8; choice 4);
    - `test_on_a_real_run_the_outcome_keeps_its_fingerprint` (choice 4);
    - `test_a_shutdown_that_fails_does_not_replace_the_exception`.
- **Unit test 3 — the drivers, by `ast`.** `TestTheDrivers` (3):
  - `test_each_ray_init_wrapper_starts_a_local_instance_then_installs_the_handler`. In
    `main`'s and `run_build`'s `_local_ray_init`: one `_real_ray_init(...)` call with
    `address="local"`, and a statement of the wrapper's own body, after it, calling
    `install_killed_handler(run, latch)`;
  - `test_the_one_exec_ends_through_finish_after_exec_registered_or_not`. Exactly one `exec(` in
    each function, inside the one `try` whose only handler is `except BaseException as exc:`.
    Its body is `finish_after_exec(run, exc, latch, ray.shutdown)` then a bare `raise`, and after
    the `try` comes `finish_after_exec(run, None, latch, ray.shutdown)`. One `exec`, inside the
    `try`, is what makes the unregistered path reach the call;
  - `test_the_shared_functions_import_nothing_from_ray_at_module_scope`.
- **Unit test 4.** `test_pipeline_adoption` (15) passes untouched, `TestTerminalState` included;
  every existing test under `RunRegistry/tests/` but `TestDrivers` is unchanged.
- **Unit test 5.** `TestDrivers`, as above.

## Deviations from the prompt

1. **The handler's first call does steps 1–4 only while the pipeline's outcome is undecided.**
   Once `finish_after_exec` has marked `latch.decided`, a signal is noted and returned from (choice
   4). The prompt makes "what is recorded" in that window this prompt's choice. This is how the
   choice is implemented, and it changes the handler's documented first call in that window.
   **IMPLEMENTATION CHOICE.**
2. **`alarm(0)` runs on every path of `finish_after_exec`, not only the signalled one.** K2 calls
   it a guard for a signal that reached Ray's handler before the driver's was installed. In that
   case the driver's handler never fired, so the latch is empty and the path is the
   not-signalled one. On a path with no alarm pending it returns 0 and changes nothing, which
   `TestNotSignalled` pins. Breakage (i) shows it working: with Ray's handler left in place, the
   probe's child printed `** cancelled an alarm with 3 s left: Ray's own signal handler ran` in 3
   of 3 runs, and none was aborted. **IMPLEMENTATION CHOICE.**
3. **The handler also guards its own `run.finish`.** A write that raises (an `OSError`, say) is
   printed and does not stop the raise; `finish_after_exec` writes the state again. The prompt
   guarded only the print. **IMPLEMENTATION CHOICE.**
4. **`finish_after_exec` returns on the not-signalled path, and the driver re-raises with a bare
   `raise`.** The prompt asks for each driver's body to shrink to "a call". It is a call and a
   `raise`. A `raise exc` inside the function would add its frame to the traceback, and "with no
   signal and no run it re-raises exactly as now" is literal only for a bare `raise` in the frame
   that caught it. **IMPLEMENTATION CHOICE.**
5. **`finish_after_exec` takes the latch, not the remembered signal's value.** It must mark the
   outcome decided in the same step as it reads the signal (`SignalLatch.decide()`), or a signal
   between the read and the mark would go to neither path. **STRUCTURALLY REQUIRED** by choice 4.
6. **Support beside the one function:** `SignalLatch`, `make_killed_handler` (the handler, built
   but not installed, so that tests call it), `KILLED_SIGNALS`, and five private helpers. README
   §4's one function is `install_killed_handler`. **IMPLEMENTATION CHOICE.**
7. **The probe compiles each driver's real `_local_ray_init` from its source**, with `ast`, rather
   than reproducing its shape, and alternates the two drivers' wrappers between runs (`scoped` for
   #1 and #3, `run_build` for #2). So breakage (i), applied to the drivers, reaches the probe.
   **IMPLEMENTATION CHOICE.**
8. **The probe gained a `phase` section and a provenance header after its first invocation.**
   Two of the first nine runs wrote `killed` 1.005 s and 1.006 s after the signal. The `phase`
   section measures why. The header prints `git HEAD` and, for the probe and both drivers,
   whether each differs from it and its SHA-256. **IMPLEMENTATION CHOICE.**
9. **`sigterm_handler_probe.txt` holds three invocations, appended, not one** (CLAUDE.md rule 6):
   - **A** (16:15–16:19), by the probe before deviation 8, on drivers whose docstrings still said
     the handler runs "within milliseconds";
   - **B** (16:22–16:26), by the committed probe, on those same drivers;
   - **C** (16:31–16:35), by the committed probe on the committed drivers. Its header shows
     `scoped_pipeline_run.py: sha256 846ec9481375c235…` and `quadsource_atol_sweep.py: sha256
     e1f6641f6fee8974…`.

   The only change to the drivers between B and C is to docstrings: three statements of the
   handler's timing, and the `terminal_state` paragraph on `ray.init`. **IMPLEMENTATION CHOICE.**
10. **The drivers' docstrings and comments changed beyond K3:** `scoped_pipeline_run.py`'s module
    docstring (item 1, and a *SIGNALS* paragraph), a comment in each driver. They describe K1, K2
    and K4. **IMPLEMENTATION CHOICE.**
11. **`_finish_calls`' docstring** says "signal handler" for "terminal handler", and the handler
    names are a module tuple `_HANDLERS` with a comment. This is the first of §3 item 5's two
    permitted changes, written out. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT` was found.

## Verification performed

### The sequence of events after a `SIGTERM`

**Before this prompt** (`8690dc6`), a registered driver blocked in `ray.wait` (audit S1–S3,
prompt 10's runs, the orchestrator's facts):

1. `register()` installs `terminal` for `SIGTERM` and `SIGINT`, before Ray.
2. `_local_ray_init` calls `ray.init(num_cpus=…, include_dashboard=False,
   ignore_reinit_error=True)` with no `address`.
   - On this machine, with ChamPBH's `ray start` cluster named in
     `/tmp/ray/ray_current_cluster`, it raised `ValueError: When connecting to an existing
     cluster, num_cpus and num_gpus must not be provided.` (the orchestrator's measurement).
   - Elsewhere it starts a local instance. `ray.init` installs `sys.exit(signum)` as the Python
     `SIGTERM` handler and absl's failure-signal handler in C, replacing `terminal`.
3. The `SIGTERM` reaches absl's handler. It prints `*** SIGTERM received` and a stack, arms a 3 s
   `SIGALRM`, and passes the signal to `sys.exit(15)`.
4. At Ray's next signal check the `SystemExit` reaches the driver's frame as `SystemExit(1)`.
   `main.py` unwinds, and its `with ShardedPool` closes the store.
5. `except SystemExit`: `run.finish(terminal_state(1), exit_code=1, fingerprint=True)`, which
   takes the fingerprint first and writes `failed`/1 after it.
6. If that is done before the alarm, the driver re-raises and exits 1, and Ray's `atexit` hook
   shuts Ray down (`resumed-a`: `failed`/1 after 2.2 s).
7. Otherwise `SIGABRT` kills it 2.78–3.06 s after the signal, before anything is written. The
   run reads `running`, then stale, and a `gcs_server` may be left behind (`resumed-b`; audit S3,
   4 of 6 aborts).
8. Unregistered, nothing is written, and the exit is 1 or `SIGABRT`. A `SIGINT` is Python's
   `KeyboardInterrupt`, recorded `failed`/1.

**After this prompt** (probe, invocation C, `second-sigterm #1`, times from the signal):

1. `register()` still installs `terminal`, for the time before Ray.
2. `_local_ray_init` calls `ray.init(address="local", …)`, which starts a fresh instance beside
   ChamPBH's (`gcs=127.0.0.1:52707 ray_current_cluster=127.0.0.1:6379`).
3. It then calls `install_killed_handler(run, latch)`. Afterwards `C SIGTERM Python+0x1e24fc
   (dladdr nearest signal_handler) | C SIGALRM SIG_DFL | Python SIGTERM
   make_killed_handler.<locals>.killed_handler | Python SIGINT
   make_killed_handler.<locals>.killed_handler`: absl is displaced, and no alarm is armed.
4. At Ray's next signal check inside `ray.wait` the handler runs on the main thread. It:
   - sets `latch.signum = 15`;
   - writes `killed`/143 with no fingerprint (`('killed', 143, False) written +0.007 s after the
     signal`);
   - prints its line;
   - raises `SystemExit(143)`.

   No banner is printed (`banner printed: 0 time(s)`).
5. It reaches the driver as `SystemExit(1)` (`t=+0.010s reached the driver: builtins.SystemExit
   code=1; latch.signum=15`). `main.py` unwinds.
6. `finish_after_exec`:
   - `decide()` returns 15;
   - `alarm(0)` returns 0;
   - `run.finish("killed", exit_code=143, fingerprint=True)`. In this variant the fingerprint
     sleeps 4 s first.
7. A second `SIGTERM` 1.5 s into that sleep prints `ignored, the run is already being stopped by
   SIGTERM` and returns. The sleep goes on.
8. The fingerprint is written (`('killed', 143, True) written +4.207 s after the signal`). Then
   `ray.shutdown()`, then `SystemExit(143)`.
9. Exit status 143, seen 5.785 s after the signal. None of the child's own processes, its
   `gcs_server` included, exists at +0.5 s, +2, +5 or +10 s.
10. Unregistered, steps 4 and 6 write nothing; the shutdown and the exit status are the same. A
    `SIGINT` follows the same steps with 130 (unit tests; Ray installs no `SIGINT` handler on a
    driver, so the probe sends only `SIGTERM`).

### The probe on real Ray

`PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/sigterm_handler_probe.py`, three invocations
appended to `sigterm_handler_probe.txt` (deviation 9).

| Invocation | Wall time | Load average (1, 5, 15 min) at start → end | Per-run 1-min load | Accepted |
|---|---|---|---|---|
| A, 16:15:35–16:19:00 | 204.8 s | (12.4, 27.1, 84.2) → (20.5, 21.6, 69.3) | 4–26 | 7 of 9 |
| B, 16:22:20–16:26:16 | 235.3 s | (9.8, 15.7, 56.8) → (6.4, 10.3, 44.5) | 4–10 | 8 of 9 |
| C, 16:31:05–16:35:01 | 236.4 s | (8.7, 8.8, 33.9) → (8.7, 8.5, 27.6) | 6–9 | 8 of 9 |

The dispatch-time load of 120–380 had fallen by then: at 16:02, before any work, `uptime` gave
127.6, 212.9, 183.9.

**Invocation C's nine runs**, pasted from its summary (nine lines):

```
plain            #1 scoped    state=killed exit_code=143 killed after 0.006 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [88387]); load 9->8; PASS
plain            #2 run_build state=killed exit_code=143 killed after 0.001 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [88627]); load 9->9; PASS
plain            #3 scoped    state=killed exit_code=143 killed after 0.007 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [89052]); load 9->8; PASS
slow-fingerprint #1 scoped    state=killed exit_code=143 killed after 0.004 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [89359]); load 9->8; PASS
slow-fingerprint #2 run_build state=killed exit_code=143 killed after 1.003 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [89798]); load 9->9; FAIL
slow-fingerprint #3 scoped    state=killed exit_code=143 killed after 0.002 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [90417]); load 8->8; PASS
second-sigterm   #1 scoped    state=killed exit_code=143 killed after 0.007 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [90887]); load 7->6; PASS
second-sigterm   #2 run_build state=killed exit_code=143 killed after 0.002 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [91344]); load 7->7; PASS
second-sigterm   #3 scoped    state=killed exit_code=143 killed after 0.007 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [91788]); load 7->8; PASS
```

"killed after" is the `status.json` mtime of the first `killed` the runner read, less the time
the child sent the signal. "survivors" are the child's own processes still existing at +10 s.

**Across all three invocations, 27 runs:**
- `killed`/143 in 27, with the fingerprint field present in 27.
- Exit status 143 in 27; no `SIGABRT`; Ray's banner printed 0 times.
- The child's own `gcs_server` seen in 27, and alive at +0.5 s in none (29 of 29 checks at
  +0.5 s are `[]`, the two `phase` children included).
- `pids killed by this probe: (none)` in all three.
- `killed` first written 0.001–0.007 s after the signal in 23 runs, and 1.003–1.006 s in 4:
  - A `plain #2` (run_build), 1.006 s;
  - A `second-sigterm #2` (run_build), 1.005 s;
  - B `plain #1` (scoped), 1.004 s;
  - C `slow-fingerprint #2` (run_build), 1.003 s.
- `killed` with the fingerprint written 0.131–9.248 s after the signal (the slow variants sleep
  4 s and 8 s first).
- Every `second-sigterm` run printed the handler's `ignored` line once (9 lines).

**The one-second criterion: failed in 4 of 27 runs, by 3–6 ms. It is not load.** As the brief
asks, the probe was re-run twice, at loads of 4–10. Each 1 s case is the handler running a whole
second after the signal. The `phase` section (B and C, the same within 0.01 s) times that
directly:

```
offset 0.10 s after entering ray.wait: SystemExit(code=1) reached the frame 0.900 s after the signal
offset 0.25 s after entering ray.wait: SystemExit(code=1) reached the frame 0.747 s after the signal
offset 0.50 s after entering ray.wait: SystemExit(code=1) reached the frame 0.499 s after the signal
offset 0.75 s after entering ray.wait: SystemExit(code=1) reached the frame 0.248 s after the signal
offset 0.90 s after entering ray.wait: SystemExit(code=1) reached the frame 0.101 s after the signal
offset 1.10 s after entering ray.wait: SystemExit(code=1) reached the frame 0.905 s after the signal
offset 1.50 s after entering ray.wait: SystemExit(code=1) reached the frame 0.504 s after the signal
offset 2.00 s after entering ray.wait: SystemExit(code=1) reached the frame 0.009 s after the signal
offset 2.05 s after entering ray.wait: SystemExit(code=1) reached the frame 0.961 s after the signal
offset 2.50 s after entering ray.wait: SystemExit(code=1) reached the frame 0.511 s after the signal
```

(invocation C). The handler runs at the next whole second of the wait.
- The probe's runs send the signal 2.0 s after the wait starts, on that boundary, so each lands
  just before it (a few ms) or just after (a whole second).
- That the boundary is Ray's `get_check_signal_interval_milliseconds` is **inferred**. The
  name, and `RAY_get_check_signal_interval_milliseconds`, are in `_raylet.so`'s strings; its
  value of 1000 ms is not read here.
- Audit S1's "within milliseconds" was measured with the same 2.0 s timer, so it is the
  early-phase case of this, not a bound.

So with this handler a `SIGTERM` in `ray.wait` is recorded up to about 1.0 s after it arrives,
plus the write. Nothing aborts in that time: no alarm is armed (C SIGALRM `SIG_DFL` after
installation, in every run). The criterion is reported as failed, not loosened, and recorded as
`[04-a-sigterm-in-ray-wait-is-handled-up-to-a-second-late]`.

### Suites

README §7's command, serially, each suite's output to a file in the agent's scratch directory, the
verdict grepped from it, at the tree of `79ab21a`:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 686 OK |
| CosmologyModels | 39 OK |
| Datastore | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 225 OK |

Measured 16:35:28–16:43:08, at 1-minute load averages of 5.9–17.5 on 10 CPUs. RunRegistry is 199 + the 26 tests of `test_killed_handler.py`; `TestDrivers` changed in place and adds none. Every other count is the dispatch baseline. No test failed, so no wall-clock test needed a re-run.

### Formatting and staging

`./venv/bin/python -m black --check` on the five touched Python files: `5 files would be left
unchanged`. Staged by explicit path; `git diff --cached --name-only` is in the commit's record
below (*State handed to the next prompt*).

## The deliberate-breakage record

The diffs were made from the working tree, which is `79ab21a`'s tree, with Python's `difflib`
and `a/`/`b/` paths. Each was applied with `git apply` and run against
`RunRegistry.tests.test_killed_handler`, `RunRegistry.tests.test_store_fingerprint.TestDrivers`
and `RunRegistry.tests.test_pipeline_adoption` (42 tests). It was then reverted with `git apply
-R`, after which `git diff` matched the saved clean state byte for byte. The harness ran in
`bash`. (i) and (iv) were also run on the probe: `--variants plain --repeat 3 --no-phase`, output
in the agent's scratch directory, not committed.

To replay one from a clean tree at `79ab21a`: save the block to `x.diff`, then `git apply
x.diff`.

**(i) The handler installed before `ray.init` instead of after**, in both drivers. Unit test 3:
`TestTheDrivers.test_each_ray_init_wrapper_starts_a_local_instance_then_installs_the_handler`
fails for both drivers (2 failures of 42). The probe, at 16:28:54–16:29:51, load 6.5 → 5.3 (Ray's
handler is back, and the state is `failed`):

```
plain            #1 scoped    state=failed exit_code=1 killed after never; fingerprint present; exit status 1; banner 2; survivors none (own gcs_server [86249]); load 7->7; FAIL
plain            #2 run_build state=failed exit_code=1 killed after never; fingerprint present; exit status 1; banner 2; survivors none (own gcs_server [86644]); load 6->6; FAIL
plain            #3 scoped    state=failed exit_code=1 killed after never; fingerprint present; exit status 1; banner 2; survivors none (own gcs_server [86854]); load 6->6; FAIL
```

After `ray.init` the C handler was `_raylet.so+0xd8fb38 = nm
__ZN4absl12lts_20230802L24AbslFailureSignalHandlerEiP9__siginfoPv` and the Python one
`init.<locals>.sigterm_handler`. `latch.signum=None`, and `SystemExit code=1` reached the driver.
The abort did **not** come: `** cancelled an alarm with 3 s left: Ray's own signal handler ran`
was printed in 3 of 3 runs (deviation 2).

```diff
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -685,13 +685,13 @@
             f"** scoped_pipeline_run: bootstrapping a local Ray instance "
             f"(num_cpus={args.cpus}, include_dashboard=False)"
         )
+        install_killed_handler(run, latch)
         context = _real_ray_init(
             address="local",
             num_cpus=args.cpus,
             include_dashboard=False,
             ignore_reinit_error=True,
         )
-        install_killed_handler(run, latch)
         return context
 
     ray.init = _local_ray_init
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -522,13 +522,13 @@
             f"** quadsource_atol_sweep --build: bootstrapping a local Ray instance "
             f"(num_cpus={args.cpus}, include_dashboard=False)"
         )
+        scoped.install_killed_handler(run, latch)
         context = _real_ray_init(
             address="local",
             num_cpus=args.cpus,
             include_dashboard=False,
             ignore_reinit_error=True,
         )
-        scoped.install_killed_handler(run, latch)
         return context
 
     ray.init = _local_ray_init
```

**(ii) The handler does not write `killed` itself.** Unit test 1 fails: 16 failures of 42.
`test_its_first_call_writes_killed_143_remembers_prints_and_raises` fails with `Lists differ: []
!= [('killed', 143, False)]`. So do the other handler, install and signalled tests that expect
that write, and `TestDrivers` for `scoped_pipeline_run.py` (`(4, 1) != (4, 2)`).

```diff
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -395,7 +395,7 @@
         code = 128 + signum
         if run is not None:
             try:
-                run.finish("killed", exit_code=code)
+                pass
             except Exception as e:  # noqa: BLE001
                 _say(stream, f"** {name}: could not record killed at once: {e!r}")
         _say(
```

**(iii) The post-`exec` function uses `terminal_state(exc.code)` when signalled.** Unit test 2
fails: 9 failures of 42. `test_it_is_killed_143_with_the_fingerprint_whatever_reaches_the_driver`
fails in all 6 sub-tests; for `SystemExit(1)` it gives `Lists differ: [('killed', 143, False),
('failed', 1, True)] != [('killed', 143, False), ('killed', 143, True)]`. Also failing:
`test_sigint_is_killed_130`, `test_a_second_signal_during_the_fingerprint_is_ignored` (`('failed',
1) != ('killed', 143)`), and `test_a_signal_inside_decide_is_a_signal_that_came_first`.

```diff
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -493,7 +493,9 @@
     try:
         if run is not None:
             try:
-                run.finish("killed", exit_code=code, fingerprint=True)
+                run.finish(
+                    terminal_state(exc.code), exit_code=exc.code, fingerprint=True
+                )
             except Exception as e:  # noqa: BLE001
                 _say(stream, f"** could not record killed with its fingerprint: {e!r}")
     finally:
```

**(iv) `ray.shutdown()` is not called.** Unit test 2 fails: 38 failures of 42, every test that
counts the shutdown calls, on both paths. The probe, at 16:29:52–16:30:50, load 5.3 → 9.9: **no
`gcs_server` was left in any run**. Without the explicit call, the exit went through Ray's
`atexit` hook, which audit S3 found left none in 9 of 9:

```
plain            #1 scoped    state=killed exit_code=143 killed after 0.001 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [87218]); load 5->8; PASS
plain            #2 run_build state=killed exit_code=143 killed after 0.004 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [87542]); load 8->8; PASS
plain            #3 scoped    state=killed exit_code=143 killed after 0.004 s; fingerprint present; exit status 143; banner 0; survivors none (own gcs_server [87771]); load 10->10; PASS
```

```diff
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -505,7 +505,7 @@
     """`shutdown()`, once. A failure is printed and goes no further: it must not replace the
     exception the caller re-raises, nor the exit status a signal gave."""
     try:
-        shutdown()
+        pass
     except Exception as e:  # noqa: BLE001
         _say(stream, f"** ray.shutdown() raised {e!r}")
 
```

**(v) A second signal raises again.** Unit test 1 fails: 10 failures and 4 errors of 42.
`test_its_second_call_writes_nothing_and_raises_nothing` errs with `SystemExit: 143`, as do
`test_installing_again_keeps_the_remembered_signal`, `test_a_print_that_fails_does_not_stop_the_raise`
and `test_once_the_outcome_is_decided_a_signal_is_only_noted`. Also failing:
`test_a_second_signal_during_the_fingerprint_is_ignored`, and choice 4's tests (8 sub-tests and
the real-`Run` one).

```diff
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -390,7 +390,7 @@
                 stream,
                 f"** {name} at {_now()}: ignored, {why}. SIGKILL stops the process at once",
             )
-            return
+            raise SystemExit(128 + signum)
         latch.signum = signum
         code = 128 + signum
         if run is not None:
```

**(vi) `address="local"` dropped from `run_build`'s real `ray.init`.** Unit test 3 fails: 1
failure of 42, `test_each_ray_init_wrapper_starts_a_local_instance_then_installs_the_handler`
(`driver='quadsource_atol_sweep.py'`: `0 != 1 : the real ray.init passes no address`).

```diff
--- a/docs/handover/quadsource_atol_sweep.py
+++ b/docs/handover/quadsource_atol_sweep.py
@@ -523,7 +523,6 @@
             f"(num_cpus={args.cpus}, include_dashboard=False)"
         )
         context = _real_ray_init(
-            address="local",
             num_cpus=args.cpus,
             include_dashboard=False,
             ignore_reinit_error=True,
```

**(vii) The post-`exec` function consults the remembered signal only for a `SystemExit`.** Unit
test 2 fails: 10 failures and 1 error of 42.
`test_it_is_killed_143_with_the_fingerprint_whatever_reaches_the_driver` fails for
`KeyboardInterrupt` and for `RuntimeError` with `Lists differ: [('killed', 143, False), ('failed',
1, True)] != [('killed', 143, False), ('killed', 143, True)]`, the `failed` the prompt names. It
also fails for no exception. Also failing: `TestNotSignalled`'s two outcome tests (3 sub-tests
each, where `decide()` is no longer called), `test_on_a_real_run_the_outcome_keeps_its_fingerprint`,
and `test_a_signal_inside_decide_is_a_signal_that_came_first` (error).

```diff
--- a/docs/gktk-remedial/scoped_pipeline_run.py
+++ b/docs/gktk-remedial/scoped_pipeline_run.py
@@ -456,7 +456,7 @@
     write its own state over the handler's `killed`, and re-raise.
     """
     try:
-        signum = latch.decide()
+        signum = latch.decide() if isinstance(exc, SystemExit) else None
     except BaseException:
         # the handler ran between decide()'s test and its assignment, and raised
         if latch.signum is None:
```

## Observations not acted on

1. **A `SIGTERM` in `ray.wait` is handled up to about a second late** (*Verification performed*).
   Ray checks for signals once a second inside a wait (the interval's name inferred from
   `_raylet.so`), so the handler, and the `killed` it writes, comes 0.001–1.006 s after the
   signal (27 runs). Shortening it means setting Ray's system config in the drivers' `ray.init`,
   which K4 kept to `address`. Opened `[04-a-sigterm-in-ray-wait-is-handled-up-to-a-second-late]`.
2. **The `status.json.tmp` window** (the orchestrator's, read; this prompt agrees it is real).
   - **The window.** The stdout `RegisteredStream` beats through `Run._update` at most every
     30 s. If the handler runs while that beat is inside `write_json_atomic`'s `with open(tmp,
     "w")`, before its `flush`, the handler's own `finish` opens and truncates the same
     `status.json.tmp`, writes `killed`, and renames it to `status.json`. The handler then
     raises. Unwinding closes the interrupted handle, whose buffered bytes, a partial JSON
     document, are written at offset 0 of that same inode, now `status.json`.
   - **Why it is real.** `json.dump` with `indent=2` runs Python bytecode, so the handler can run
     inside it; after the `flush` there is nothing left to write.
   - **What it costs.** `finish_after_exec`'s second `finish` reads the file as `{}` and rewrites
     it valid, without `pid`, `stage` or `units_*`. A `SIGKILL` in between would leave it
     unparseable.
   - Not measured; not fixed, since `RunRegistry` is outside this prompt (§6). Opened
     `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`.
3. **A `SIGTERM` while `ray.init` itself runs** (read, not measured).
   - **The window.** Ray installs `sys.exit(signum)` at the start of `ray.init` (`worker.py`
     L1498–1502), and the driver's handler is installed only after it returns: about 2.3 s on
     this machine (the orchestrator's measurement). A signal then goes to Ray's handler.
   - **What happens.** The latch stays empty, and `finish_after_exec` takes the not-signalled
     path: `alarm(0)`, then `terminal_state(exc.code)`. That is `killed`/15 if the `SystemExit`
     arrives as 15, and `failed`/1 if Ray's check converts it to 1. The exit is that code, not
     143. Which one happens is not established.
   - Opened `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]`.
4. **`run_child` and `sweep`** (K1, K4: unchanged, `ast.unparse` equal to `8690dc6`'s).
   - **`run_child` shares both defects.** It installs no handler, so Ray's handler governs it: a
     `SIGTERM` in `ray.wait` reaches it as `SystemExit(1)`, with absl's alarm armed. And its
     `_local_ray_init` passes no `address`, so on this machine its `ray.init(num_cpus=…)` would
     raise the `ValueError` the orchestrator measured.
   - **`sweep`'s `terminal` works as written**: `sweep` never calls `ray.init`, so nothing
     replaces it.
   - Opened `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]`. It does
     not touch A3 v2's launch, which is `--build`.
5. **A signal between the driver's `except BaseException as exc:` and the first instruction of
   `finish_after_exec`'s `try`** (read).
   - **What happens.** The handler writes `killed`/143 and raises `SystemExit(143)` out of the
     driver's `except`. The process exits 143 with no fingerprint and no explicit
     `ray.shutdown()`; Ray's `atexit` hook still runs it.
   - **Why no issue.** The record is right, and the window is a few bytecodes.
6. **`TestTerminalState.test_a_sigterm_that_arrives_as_rays_systemexit_is_killed_not_failed`**
   keeps its name and its docstring ("a SIGTERM reaches the driver as `SystemExit(15)` — observed
   in this prompt's demonstration"). Both record the belief K3 corrects. It passes, and is not
   renamed (K3; README §5 rule 7).
7. **`include_dashboard=False` still starts a `dashboard/dashboard.py`** child, with `--port=8265
   --port-retries=50`, in every probe run (the own-process lists). ChamPBH's head serves its
   dashboard on 8265, and the probe's instances started regardless. Ray's behaviour, noted only.

## State handed to the next prompt

- **Issues.**
  - `[10-a-sigterm-…]` is closed on the `datastore-integrity` board's §4, and its index row is
    deleted.
  - Opened on this board's §3.2:
    - `[04-a-sigterm-in-ray-wait-is-handled-up-to-a-second-late]`;
    - `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`;
    - `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]`;
    - `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]`.
  - Index: **105 open** (102 − 1 + 4).
- **For 05**, which also edits `run_build`: this prompt changed its `_local_ray_init` and its
  `exec` lines. It did not change the `sys.argv` list that carries `--no-prune-unvalidated`.
- **For 07's rehearsal and A3 v2's launch on the macstudio** (K4):
  - Both drivers now start a fresh local instance whatever `ray start` has left in
    `/tmp/ray/ray_current_cluster`. They never attach to another cluster, and are not refused by
    one.
  - A `SIGTERM` to a registered `--build` in a sharded stage is recorded `killed`/143 within about
    a second, then gains its fingerprint, and the driver exits 143 after `ray.shutdown()`. The
    handler's line, `** SIGTERM at …: recorded killed, exit code 143; …`, is how the
    interruption shows in `stderr.log`. Ray's `*** SIGTERM received` banner no longer prints.
  - A `SIGINT` is recorded `killed`/130.
- **Suites:** AdaptiveLevin 32, ComputeTargets 686, CosmologyModels 39, Datastore 598, LiouvilleGreen 148 (1 skipped), RunRegistry 225, all OK.
- **Staged:** `RunRegistry/tests/test_killed_handler.py`, `RunRegistry/tests/test_store_fingerprint.py`, `docs/OPEN_ISSUES.md`, `docs/a3-v2-readiness/sigterm_handler_probe.py`, `docs/a3-v2-readiness/sigterm_handler_probe.txt`, `docs/gktk-remedial/scoped_pipeline_run.py`, `docs/handover/quadsource_atol_sweep.py`, `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md`, `prompts/a3-v2-readiness/logs/04-record-a-sigterm-as-killed.md`, `prompts/datastore-integrity/IMPLEMENTATION_STATE.md` (ten files).
