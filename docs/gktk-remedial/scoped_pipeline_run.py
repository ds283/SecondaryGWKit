"""Run main.py's pipeline on a scoped wavenumber sample, against a locally bootstrapped Ray
instance and a fresh datastore.

Copied verbatim from docs/source-remediation-verification/scoped_pipeline_run.py for prompt 13 of
prompts/GkTk-remedial (Layer 2 of this campaign's verification pass), with **one** change: the two
wavenumber-grid literals it substitutes are the ones main.py carries today. Since `f17f2d4`
main.py spells its grids

    np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)
    np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)

so the original script's exact-text match (`..., 50)`) finds zero occurrences and refuses to run.
That file belongs to the source-remediation campaign and is not edited here; this copy is the
GkTk-remedial campaign's driver. See prompts/GkTk-remedial/IMPLEMENTATION_STATE.md
`[13-scoped-run-driver-k-grid-literal]`.

Written for prompt 12 of prompts/source-remediation (Layer 2 of the verification pass). The
pipeline itself is main.py's: this script does not reimplement any stage. main.py cannot be
imported (it parses sys.argv, opens a Ray connection and a ShardedPool at module scope, and then
runs the pipeline), so this driver

  1. patches ray.init so that main.py's own `ray.init(address=...)` bootstraps a local instance
     with a chosen CPU count (`address="local"`) instead of attaching to a cluster, even one
     that `ray start` has left running here, and installs this driver's signal handler once Ray
     is up (see "SIGNALS" below);
  2. optionally filters config.model_list.build_model_list's return value, so that a run can be
     restricted to one cosmology;
  3. reads main.py's source, substitutes the two hardcoded 50-point wavenumber grids
     (`np.logspace(np.log10(1e5), np.log10(3e8), 50)`, main.py's source and response k samples)
     for a name supplied in the execution globals, and exec()s the result.

Only those three things change; every stage, tag, tolerance and work-queue parameter is main.py's.
The substitution is textual and exact, and the script prints what it replaced and how many times.

Usage (all main.py options after `--` are passed through verbatim):

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/scoped_pipeline_run.py \
        --k-min 1e5 --k-max 1e7 --k-count 7 --cpus 10 --models LambdaCDM \
        -- --database /path/to/fresh.sqlite --job-name label --shards 4 \
           --zend 1e4 --source-samples-log10z 100

The datastore path must not already exist; this script refuses to run otherwise, so that no
existing datastore can be overwritten or migrated.

REGISTERING THE RUN. A pipeline run takes hours -- the A3 baseline ran for 10 h 33 m -- and
`prompts/run-registry` README §0 is a list of what that costs when nothing on disk says the job
exists. With `--register <slug> --purpose "..."` this script writes a `runregistrykit` manifest
**before** the pipeline starts, keeps `status.json` current while it runs, tees its output into
the run directory and records a terminal state on exit. Without the flag nothing changes at all.

**Registration is not a fourth change to main.py.** The three changes above are changes to
*main.py*: its `ray.init`, its model list and its two grid literals. Registration adds nothing to
the execution namespace, alters no argument in `sys.argv`, substitutes no further text and reads
no result -- `main.py` runs byte-identically under `--register` and without it. What it touches is
this driver's own process: a manifest written before the exec, `sys.stdout` and `sys.stderr`
wrapped by a proxy that passes every write through unaltered and copies it into the run
directory, and a terminal state written after it.

**What the heartbeat means, exactly.** The work happens in Ray actors, so a heartbeat from this
process cannot assert that any actor is progressing. It is refreshed when the pipeline writes a
line to stdout, at most once every `BEAT_MIN_INTERVAL` seconds. `RayWorkPool` prints its progress
line only after `ray.wait` has returned completed work, so a fresh heartbeat does mean *work
completed recently* -- it is stronger than "the parent is alive", and a parent blocked for ever
inside `ray.get` prints nothing and goes stale, which is the reading we want. The converse does
**not** hold: a stage that prints nothing for longer than the staleness window -- one long work
item, or a background-model build -- is reported stale while it is legitimately working. Stale
means "go and look", not "dead", and this sentence is in the manifest as `heartbeat_means`.

SIGNALS. `ray.init` replaces any `SIGTERM` handler installed before it (`terminal`, from
`register()`, covers only the time before Ray is up). So the `ray.init` wrapper installs
`install_killed_handler`'s handler for `SIGTERM` and `SIGINT` straight after Ray starts, registered
or not. A signal is then recorded `killed`, `128 + signum`, as soon as the handler runs, which in
`ray.wait` is at Ray's next check for signals, up to about a second later
(`docs/a3-v2-readiness/sigterm_handler_probe.txt`); after `main.py` unwinds, `finish_after_exec`
adds the fingerprint, shuts Ray down and exits with `128 + signum`
(`docs/a3-v2-readiness-audit.md` S1–S4, `prompts/a3-v2-readiness` prompt 04).
"""

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np


class Refusal(RuntimeError):
    """A check that stops this driver before any work starts: an existing store, a path that is
    not a directory, a store a registered run already names. The `__main__` block prints it as one
    `!!` line on stderr and exits 1, with no traceback. It is a `RuntimeError` on purpose, so that
    every caller and test that catches `RuntimeError` still does. A failure during the work, and an
    internal invariant that does not hold, are plain `RuntimeError`s and keep their traceback.
    """


REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"

# the exact text of both hardcoded wavenumber grids in main.py's driver block (:3094, :3106)
K_GRID_LITERALS = (
    "np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)",
    "np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)",
)
K_GRID_NAME = "SCOPED_K_SAMPLE"

# =================================================================================================
# registration. Everything from here to `register()` is inert unless --register is given, and none
# of it is a change to main.py: see "REGISTERING THE RUN" in the module docstring.

HEARTBEAT_MEANS = (
    "Refreshed when the pipeline writes a line to stdout, at most once per "
    "{interval:.0f} s, and the line it came from is the `stage` in status.json. RayWorkPool "
    "prints only after ray.wait has returned completed work, so a fresh heartbeat means work "
    "completed recently -- stronger than 'the parent process is alive', and a parent blocked "
    "for ever inside ray.get prints nothing and goes stale. It does NOT mean any Ray actor is "
    "progressing, and the converse does not hold: a stage that prints nothing for longer than "
    "the staleness window (one long work item, a background-model build) is reported stale "
    "while it is legitimately working. Stale means go and look."
)

# How often the heartbeat may be rewritten. status.json is fsynced and renamed on every beat, so
# this is throttled against main.py's per-batch progress lines rather than written per line; it is
# two orders below runregistrykit's 900 s staleness window.
BEAT_MIN_INTERVAL = 30.0

# main.py's and RayWorkPool's own progress vocabulary, which this file only reads.
_STAGE_BANNER = re.compile(r"^\s*(?:\*\*|>>)\s+(\S.*?)\s*$")
_QUEUE_PROGRESS = re.compile(
    r"(\d+)/(\d+)\s+work items remaining\s*=\s*([\d.]+)%\s+complete"
)
_QUEUE_COMPLETE = re.compile(r"ALL WORK ITEMS COMPLETE")


class StageTracker:
    """Where the pipeline has got to, read off its own output.

    The pipeline reports progress per stage and per work queue and never as one total, so there
    is no honest denominator to put in `units_done`/`units_total` and this reports a sentence
    instead. The sentence is in the form the hand-written
    `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` recorded by hand after the fact
    -- "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31% (1 of 13 work items remaining)" -- because
    reconstructing that from a stdout file is the work the registry exists to save.
    """

    def __init__(self):
        self.title = None
        self.stage = None

    def feed(self, line) -> bool:
        """Read one line; return True if the stage changed."""
        before = self.stage
        banner = _STAGE_BANNER.match(line)
        if banner is not None:
            self.title = banner.group(1)
            self.stage = self.title
        elif self.title is not None:
            progress = _QUEUE_PROGRESS.search(line)
            if progress is not None:
                remaining, total, percent = progress.groups()
                self.stage = (
                    f"{self.title}, {percent}% "
                    f"({remaining} of {total} work items remaining)"
                )
            elif _QUEUE_COMPLETE.search(line):
                self.stage = f"{self.title}, all work items complete"
        return self.stage != before


class RegisteredStream:
    """`sys.stdout`, with a copy into the run directory and a heartbeat off the back of it.

    Every write is passed through to the real stream unaltered and unconditionally; the tee, the
    stage tracking and the beat are wrapped so that a failure in any of them cannot cost the
    pipeline a line of output or an exception it did not raise. Attribute lookups that are not
    `write`, `writelines` or `flush` go to the real stream, so `fileno`, `isatty`, `encoding` and
    `buffer` are whatever they were.
    """

    def __init__(
        self,
        stream,
        run,
        tracker,
        copy_to=None,
        interval=BEAT_MIN_INTERVAL,
        beats=True,
    ):
        self._stream = stream
        self._run = run
        self._tracker = tracker
        self._interval = interval
        self._beats = beats
        self._pending = ""
        self._last_beat = 0.0
        self._copy = None
        if copy_to is not None:
            try:
                self._copy = open(copy_to, "a", buffering=1)
            except OSError:
                self._copy = None

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, text):
        written = self._stream.write(text)
        # the output is the pipeline's; the bookkeeping is ours, and never costs it an exception
        try:
            self._observe(text)
        except Exception:  # noqa: BLE001
            pass
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self._stream.flush()
        if self._copy is not None:
            try:
                self._copy.flush()
            except OSError:
                pass

    def close_copy(self):
        """Close the run directory's copy, leaving the real stream open.

        Called only on the clean path, so that a failing run's traceback -- printed by the
        interpreter after `main()` unwinds -- still reaches `stderr.log`. A killed or failed run
        loses nothing by not reaching it: the copy is line-buffered, and the kernel closes what
        is left."""
        copy, self._copy = self._copy, None
        if copy is not None:
            try:
                copy.close()
            except OSError:
                pass

    def _observe(self, text):
        if self._copy is not None:
            # a tee that has failed once is dropped rather than retried, and must not cost the
            # stage tracking or the heartbeat below
            try:
                self._copy.write(text)
            except Exception:  # noqa: BLE001
                self._copy = None
        if not self._beats:
            return
        self._pending += text
        if "\n" not in self._pending:
            return
        *lines, self._pending = self._pending.split("\n")
        changed = False
        for line in lines:
            changed = self._tracker.feed(line) or changed
        now = time.monotonic()
        if changed or now - self._last_beat >= self._interval:
            self._last_beat = now
            self._run.heartbeat(stage=self._tracker.stage)


def terminal_state(code) -> str:
    """The state to record for a `SystemExit` out of the pipeline **that no signal caused**.

    It is not how a `SIGTERM` is recorded. This docstring used to say that once Ray is up a
    `SIGTERM` reaches the driver as `SystemExit(15)`, after Ray's own shutdown has run. Measured,
    that is false (`docs/a3-v2-readiness-audit.md` S1–S3):

    - `ray.init` installs `sys.exit(signum)` as the Python-level `SIGTERM` handler at its start,
      and absl's failure-signal handler at the C level, so a driver handler installed before it
      never runs once Ray is up (S1);
    - with Ray's handlers in place, a `SIGTERM` while the driver waits in `ray.wait` reaches it as
      `SystemExit(1)`, not 15, with no shutdown run, and absl arms a 3 s `SIGALRM` that kills the
      process by `SIGABRT` 2.78–3.06 s after the signal unless `signal.alarm(0)` cancels it (S1,
      S2); an exit that skips `ray.shutdown()` can leave a `gcs_server` behind (S3);
    - a `SystemExit` that any Python handler raises inside `ray.wait` or `ray.get` arrives as
      `SystemExit(1)`, so the exit code cannot say that a signal came (S1).

    So since `a3-v2-readiness` prompt 04 a signal is recorded by the handler that
    `install_killed_handler` puts in place **after** `ray.init` (it writes `killed`, `128 +
    signum` at once), and `finish_after_exec` keeps that record whatever exception then reaches
    the driver. This function is used only when that handler did not fire.

    Its mapping is kept: `SystemExit(15)` is still `killed`. Ray's own handler, `sys.exit(signum)`,
    is installed at the start of `ray.init` and the driver's only after it returns, so a `SIGTERM`
    that lands while `ray.init` runs can still reach the driver as `SystemExit(15)` (read, not
    measured). The one ambiguity is a job that deliberately exits with status 15, which `main.py`
    never does — it exits 0, or with a traceback.
    """
    if code is None:
        return "done"
    if code == int(signal.SIGTERM):
        return "killed"
    return "done" if code == 0 else "failed"


# =================================================================================================
# signals once Ray is up (a3-v2-readiness prompt 04). Nothing here imports `ray`: the driver hands
# in `ray.shutdown`, and a test hands in a stand-in.

# the signals the handler takes. Ray's `ray.init` installs only SIGTERM on a driver; SIGINT would
# otherwise be Python's KeyboardInterrupt, recorded `failed`.
KILLED_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class SignalLatch:
    """Where the handler remembers the signal that ended the pipeline, for the driver to read.

    The exit code cannot carry it: a `SystemExit(128 + signum)` raised inside `ray.wait` reaches
    the driver as `SystemExit(1)` (audit S1). One latch serves a whole run. Installing the handler
    again, as a second `ray.init` would (`ignore_reinit_error=True`), keeps it.

    - `signum` is the first signal that reached the handler **before the pipeline's own outcome
      reached the driver**, or `None`. It is set first, before anything else the handler does.
    - `decided` is set by `finish_after_exec` when the pipeline's own outcome reached it first.
      From then on a signal is noted and does not change the record (`finish_after_exec`).
    """

    def __init__(self):
        self.signum = None
        self.decided = False

    def decide(self):
        """Mark the pipeline's own outcome as decided, unless a signal came first.

        Returns the remembered signal, or `None` when the outcome is the pipeline's. A signal
        that lands between the test and the assignment runs the handler while `decided` is still
        false, so the handler raises out of this call with `signum` set: the caller treats that as
        a signal that came first."""
        if self.signum is None:
            self.decided = True
        return self.signum


def _say(stream, text):
    """One line to `stream` (by default `sys.stderr` as it is when the line is written).

    Never raises. A handler that prints while the main thread is inside a write to the same
    buffered stream gets CPython's `RuntimeError: reentrant call`; a line lost that way must not
    cost the handler the steps after it."""
    try:
        print(text, file=sys.stderr if stream is None else stream, flush=True)
    except Exception:  # noqa: BLE001
        pass


def _signal_name(signum):
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def make_killed_handler(run, latch, stream=None):
    """The handler `install_killed_handler` installs, built but not installed (tests call it).

    `run` is the registered `Run`, or `None` for an unregistered run; `latch` is the run's
    `SignalLatch`.

    **Its first call, before the pipeline's outcome is decided:**

    1. remembers the signal in `latch.signum`, first, so that nothing after it can lose it;
    2. writes `run.finish("killed", exit_code=128 + signum)` when there is a run: no fingerprint,
       because the pool is still open, and at once, because nothing after it is guaranteed. A
       failure to write is printed, and does not stop step 4: `finish_after_exec` writes the
       state again;
    3. prints one line of its own to stderr naming the signal and the time, since Ray's
       `*** SIGTERM received` banner no longer prints once this handler is installed. A failure
       to print never stops step 4;
    4. raises `SystemExit(128 + signum)`, which unwinds `main.py`, its `with ShardedPool` included.
       Raised inside `ray.wait` or `ray.get` it reaches the driver as `SystemExit(1)` (audit S1),
       which is why step 1 exists.

    Inside `ray.wait` the handler runs when Ray next checks for signals, which
    `sigterm_handler_probe.txt` measured at the next whole second of the wait: from a few
    milliseconds to about a second after the signal.

    **Any other call** — a second signal during the cleanup, or a signal after the pipeline's own
    outcome reached the driver (`SignalLatch.decided`) — prints one line and returns, raising
    nothing and writing nothing. `SIGKILL` remains for a person who means it.
    """

    def killed_handler(signum, _frame):
        name = _signal_name(signum)
        if latch.signum is not None or latch.decided:
            why = (
                f"the run is already being stopped by {_signal_name(latch.signum)}"
                if latch.signum is not None
                else "the pipeline had already ended; its own outcome is being recorded"
            )
            _say(
                stream,
                f"** {name} at {_now()}: ignored, {why}. SIGKILL stops the process at once",
            )
            return
        latch.signum = signum
        code = 128 + signum
        if run is not None:
            try:
                run.finish("killed", exit_code=code)
            except Exception as e:  # noqa: BLE001
                _say(stream, f"** {name}: could not record killed at once: {e!r}")
        _say(
            stream,
            f"** {name} at {_now()}: "
            + (f"recorded killed, exit code {code}; " if run is not None else "")
            + f"stopping the pipeline; the process will exit with status {code}",
        )
        raise SystemExit(code)

    return killed_handler


def install_killed_handler(run, latch, stream=None):
    """Install the handler for `SIGTERM` and `SIGINT`. Call it **after** `ray.init` returns.

    The one function both drivers call from their `ray.init` wrappers (`a3-v2-readiness` README
    §4). `ray.init` installs Ray's own `SIGTERM` handlers, `sys.exit(signum)` in Python and absl's
    failure-signal handler in C. A Python handler installed after it replaces both at the C level,
    runs on the main thread inside `ray.wait` and `ray.get`, prints no banner and arms no alarm
    (audit S1); inside a wait it runs at Ray's next check for signals, up to about a second after
    the signal (`docs/a3-v2-readiness/sigterm_handler_probe.txt`). It is installed whether or not
    the run is registered: `run` decides only whether a record is written. Returns the handler.
    """
    handler = make_killed_handler(run, latch, stream=stream)
    for signum in KILLED_SIGNALS:
        signal.signal(signum, handler)
    return handler


def finish_after_exec(run, exc, latch, shutdown, *, alarm=signal.alarm, stream=None):
    """What a driver does once `main.py`'s `exec` is over, with Ray up or not.

    `run` is the `Run` or `None`; `exc` the exception that ended the `exec`, or `None` if it
    returned; `latch` the run's `SignalLatch`; `shutdown` is `ray.shutdown` in a driver.

    **If the handler fired first** (`latch.signum`), whatever `exc` is — `SystemExit(1)` out of a
    Ray wait, or a `RuntimeError` or `KeyboardInterrupt` from elsewhere — the run was killed:

    1. `alarm(0)`, which cancels absl's abort if Ray's handler saw a signal first (audit S2);
    2. `run.finish("killed", exit_code=128 + signum, fingerprint=True)`, best-effort: the
       handler has written the state, `main.py`'s `with ShardedPool` has unwound, so the store
       is closed;
    3. `shutdown()`: every exit through it left no `gcs_server` (audit S3);
    4. raises `SystemExit(128 + signum)`.

    **Otherwise** the pipeline's own outcome is recorded, exactly as before this function existed:
    `done`/0 for a return, `terminal_state(exc.code)` for a `SystemExit`, `failed`/1 for anything
    else, each with the fingerprint. `alarm(0)` runs first here too (it cancels nothing unless
    Ray's own handler saw a signal inside `ray.init`), and `shutdown()` after the record. It then
    **returns**, and the caller re-raises `exc` with a bare `raise`, so the traceback is exactly
    the one it always was.

    **A signal once the pipeline's outcome is decided** — during this function's own fingerprint,
    for instance — is noted by the handler and changes nothing: the record is the pipeline's
    outcome with its fingerprint, and the exit is the pipeline's. This is prompt 04's choice for
    that window; without it `Run.finish` would catch the handler's `SystemExit` mid-fingerprint,
    write its own state over the handler's `killed`, and re-raise.
    """
    try:
        signum = latch.decide()
    except BaseException:
        # the handler ran between decide()'s test and its assignment, and raised
        if latch.signum is None:
            raise
        signum = latch.signum

    remaining = alarm(0)
    if remaining:
        _say(
            stream,
            f"** cancelled an alarm with {remaining} s left: Ray's own signal handler ran",
        )

    if signum is None:
        if run is not None:
            if exc is None:
                run.finish("done", exit_code=0, fingerprint=True)
            elif isinstance(exc, SystemExit):
                run.finish(
                    terminal_state(exc.code), exit_code=exc.code, fingerprint=True
                )
            else:
                run.finish("failed", exit_code=1, fingerprint=True)
        _shut_down(shutdown, stream)
        return

    code = 128 + signum
    if exc is not None:
        _say(
            stream,
            f"** {_signal_name(signum)}: the pipeline ended with "
            f"{type(exc).__name__}{_code_text(exc)}; recording killed, exit code {code}",
        )
    try:
        if run is not None:
            try:
                run.finish("killed", exit_code=code, fingerprint=True)
            except Exception as e:  # noqa: BLE001
                _say(stream, f"** could not record killed with its fingerprint: {e!r}")
    finally:
        _shut_down(shutdown, stream)
    raise SystemExit(code)


def _shut_down(shutdown, stream):
    """`shutdown()`, once. A failure is printed and goes no further: it must not replace the
    exception the caller re-raises, nor the exit status a signal gave."""
    try:
        shutdown()
    except Exception as e:  # noqa: BLE001
        _say(stream, f"** ray.shutdown() raised {e!r}")


def _code_text(exc):
    return f"({exc.code!r})" if isinstance(exc, SystemExit) else ""


def register(args, main_args, k_sample, database, runs_root=None):
    """Write the manifest, take over both output streams, and arrange a terminal state.
    Returns the `Run`.

    Called **before** the pipeline starts -- before Ray is bootstrapped and before main.py's
    source is read -- so that a crash in the first instant still leaves a record of what was
    running and what it was for.

    ``runs_root`` is passed to `runregistrykit.begin` as its ``root``: `None`, the default, is
    the runs root `runregistrykit.toml` names (`var/runs`); a test passes a temporary one.
    """
    try:
        import runregistrykit
    except ImportError as exc:  # pragma: no cover -- an operator-facing message
        raise Refusal(
            "scoped_pipeline_run: --register needs the runregistrykit package, which is "
            "missing from this venv; install requirements.txt into it "
            "(./venv/bin/pip install -r requirements.txt)"
        ) from exc

    models = (
        ", ".join(args.models) if args.models else "every model in config.model_list"
    )
    scope = (
        f"{len(k_sample)} log-spaced k over {args.k_min:g}-{args.k_max:g} /Mpc, {models}, "
        f"{args.cpus} cpus; main.py args: {' '.join(main_args)}"
    )
    try:
        run = runregistrykit.begin(
            campaign=args.campaign,
            prompt=args.prompt,
            slug=args.register,
            purpose=args.purpose,
            script=__file__,
            # README §0.2: this job's results live in its datastore, and `results` is the field that
            # names the durable thing they live in -- which is what a resume needs. It is deliberately
            # *not* `checkpoint`: that field is the unit ledger record()/known() own, this run has no
            # ledger and records no units, and a run that named its datastore there would be one
            # record() call away from appending JSON-Lines to a SQLite file.
            results=str(database) if database is not None else None,
            scope=scope,
            heartbeat_means=HEARTBEAT_MEANS.format(interval=BEAT_MIN_INTERVAL),
            root=runs_root,
        )
    except runregistrykit.ConfigError:
        raise
    except RuntimeError as e:
        raise Refusal(str(e)) from e

    def terminal(signum, _frame):
        """Record the death, then die exactly as the default handler would have."""
        run.finish("killed", exit_code=128 + signum)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, terminal)

    tracker = StageTracker()
    sys.stdout = RegisteredStream(
        sys.stdout, run, tracker, copy_to=run.stdout_path, interval=BEAT_MIN_INTERVAL
    )
    # stderr is copied but does not beat and carries no stage: it is where the traceback of a
    # failed run goes, and a run directory that cannot say why the job died is half a record.
    # Ray's own periodic warnings land here too, which is exactly why they must not count as
    # progress -- the heartbeat stays on the stream that only prints when work completes.
    sys.stderr = RegisteredStream(
        sys.stderr, run, StageTracker(), copy_to=run.stderr_path, beats=False
    )
    print(
        f"** scoped_pipeline_run: registered as {run.id}\n"
        f"**   manifest {run.manifest_path}\n"
        f"**   list it with: ./venv/bin/python -m runregistrykit list"
    )
    return run


def main():
    parser = argparse.ArgumentParser(
        description="run main.py's pipeline on a scoped wavenumber sample"
    )
    parser.add_argument("--k-min", type=float, required=True)
    parser.add_argument("--k-max", type=float, required=True)
    parser.add_argument("--k-count", type=int, required=True)
    parser.add_argument("--cpus", type=int, default=10)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="restrict config.model_list.build_model_list to these labels",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="permit an existing datastore (for a resume/lookup test); the pipeline only adds "
        "work that is missing, it never migrates or rewrites existing rows",
    )
    parser.add_argument(
        "--register",
        type=str,
        default=None,
        metavar="SLUG",
        help="record this run in var/runs/ through runregistrykit, under this slug. Without it "
        "nothing is registered and the run behaves exactly as it always has",
    )
    parser.add_argument(
        "--purpose",
        type=str,
        default=None,
        help="with --register: one line a stranger can read saying what this run is for",
    )
    parser.add_argument(
        "--campaign",
        type=str,
        default="gktk-remedial",
        help="with --register: the campaign that owns this run",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="-",
        help="with --register: the prompt that owns this run, if one does",
    )
    parser.add_argument(
        "main_args",
        nargs=argparse.REMAINDER,
        help="arguments for main.py, after a bare --",
    )
    args = parser.parse_args()
    if args.register and not args.purpose:
        parser.error(
            "--register needs --purpose: a run nobody can identify from disk is the thing the "
            "registry exists to prevent"
        )

    main_args = list(args.main_args)
    if main_args and main_args[0] == "--":
        main_args = main_args[1:]

    # refuse to touch an existing datastore
    db = None
    if "--database" in main_args:
        db = Path(main_args[main_args.index("--database") + 1])
        if db.exists() and not args.allow_existing:
            raise Refusal(
                f"scoped_pipeline_run: datastore {db} already exists; refusing to reuse, "
                "overwrite or migrate it. Choose a path that does not exist."
            )
        if not db.parent.is_dir():
            raise Refusal(
                f"scoped_pipeline_run: {db.parent} is not a directory, so {db} is not writable"
            )

    k_sample = np.logspace(
        np.log10(args.k_min), np.log10(args.k_max), args.k_count
    ).tolist()
    print(f"** scoped wavenumber sample ({len(k_sample)} modes, 1/Mpc):")
    for k in k_sample:
        print(f"     {k:.6g}")

    # (R) register the run, before anything long starts. This is not one of the three changes
    #     below: it changes nothing about main.py, only about this driver's own process.
    run = register(args, main_args, k_sample, db) if args.register else None

    # (1) bootstrap Ray locally rather than attaching to a cluster. `address="local"` starts a
    #     fresh instance even where `ray start` has left another cluster's address in
    #     /tmp/ray/ray_current_cluster, which a bare call would resolve to. Once it is up, the
    #     driver's own signal handler replaces Ray's (`install_killed_handler`), whether or not
    #     the run is registered.
    import ray

    _real_ray_init = ray.init
    latch = SignalLatch()

    def _local_ray_init(*_args, **_kwargs):
        print(
            f"** scoped_pipeline_run: bootstrapping a local Ray instance "
            f"(num_cpus={args.cpus}, include_dashboard=False)"
        )
        context = _real_ray_init(
            address="local",
            num_cpus=args.cpus,
            include_dashboard=False,
            ignore_reinit_error=True,
        )
        install_killed_handler(run, latch)
        return context

    ray.init = _local_ray_init

    # (2) optionally restrict the model list
    if args.models is not None:
        import config.model_list as model_list_module

        _real_build_model_list = model_list_module.build_model_list

        def _filtered_build_model_list(pool, units):
            models = _real_build_model_list(pool, units)
            kept = [m for m in models if m["label"] in args.models]
            if len(kept) == 0:
                raise RuntimeError(
                    f"scoped_pipeline_run: no model in {[m['label'] for m in models]} "
                    f"matches {args.models}"
                )
            print(
                f"** scoped_pipeline_run: running models {[m['label'] for m in kept]}"
            )
            return kept

        model_list_module.build_model_list = _filtered_build_model_list

    # (3) substitute the wavenumber grids and run
    source = MAIN_PY.read_text()
    for literal in K_GRID_LITERALS:
        count = source.count(literal)
        if count != 1:
            raise RuntimeError(
                f"scoped_pipeline_run: expected exactly 1 occurrence of "
                f"{literal!r} in main.py, found {count}"
            )
        source = source.replace(literal, K_GRID_NAME)
        print(
            f"** scoped_pipeline_run: replaced {count} occurrence of {literal!r} "
            f"in main.py with {K_GRID_NAME}"
        )

    sys.argv = ["main.py"] + main_args
    print(f"** scoped_pipeline_run: main.py argv = {sys.argv[1:]}")

    namespace = {
        "__name__": "__main__",
        "__file__": str(MAIN_PY),
        K_GRID_NAME: k_sample,
    }
    # the exec, registered or not, with a terminal state written after it when there is a run,
    # and Ray shut down either way (`finish_after_exec`). A SIGKILL never reaches here, which is
    # why the registry reports a run that still says `running` with a stale heartbeat rather than
    # inferring a death it did not see.
    try:
        exec(compile(source, str(MAIN_PY), "exec"), namespace)
    except BaseException as exc:
        # raises SystemExit(128 + signum) if a signal ended the pipeline; otherwise records its
        # outcome and returns, and this re-raises with the copies still open: the traceback the
        # interpreter is about to print is the one thing a failed run's directory most needs, and
        # it is printed after this frame unwinds. The kernel closes the copies when the process
        # goes.
        finish_after_exec(run, exc, latch, ray.shutdown)
        raise
    finish_after_exec(run, None, latch, ray.shutdown)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, RegisteredStream):
            stream.close_copy()


if __name__ == "__main__":
    try:
        main()
    except Refusal as exc:
        sys.stdout.flush()
        print(f"!! {exc}", file=sys.stderr)
        sys.exit(1)
