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
     with a chosen CPU count instead of attaching to a cluster that does not exist here;
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
           --zend 1e4 --source-samples-log10z 100 --no-prune-unvalidated

The datastore path must not already exist; this script refuses to run otherwise, so that no
existing datastore can be overwritten or migrated.

REGISTERING THE RUN. A pipeline run takes hours -- the A3 baseline ran for 10 h 33 m -- and
`prompts/run-registry` README §0 is a list of what that costs when nothing on disk says the job
exists. With `--register <slug> --purpose "..."` this script writes a `RunRegistry` manifest
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
"""

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np

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
# two orders below RunRegistry's 900 s staleness window.
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
    """The state to record for a `SystemExit` out of the pipeline.

    `ray.init` **replaces** this script's `SIGTERM` handler with its own, which is

        def sigterm_handler(signum, frame):
            sys.exit(signum)

    (`ray/_private/worker.py:1498`, installed only on the main thread). So once Ray is up, a
    `SIGTERM` reaches us as `SystemExit(15)` — after Ray's own shutdown has run, which is why the
    A3 baseline's four shards passed `PRAGMA integrity_check` when it was stopped that way.
    Recording that as `failed` would say something untrue about a run that was deliberately
    interrupted; `killed` is what it was. The one ambiguity is a job that deliberately exits with
    status 15, which `main.py` never does — it exits 0, or with a traceback.
    """
    if code is None:
        return "done"
    if code == int(signal.SIGTERM):
        return "killed"
    return "done" if code == 0 else "failed"


def register(args, main_args, k_sample, database):
    """Write the manifest, take over both output streams, and arrange a terminal state.
    Returns the `Run`.

    Called **before** the pipeline starts -- before Ray is bootstrapped and before main.py's
    source is read -- so that a crash in the first instant still leaves a record of what was
    running and what it was for.
    """
    try:
        import RunRegistry
    except ImportError as exc:  # pragma: no cover -- an operator-facing message
        raise RuntimeError(
            "scoped_pipeline_run: --register needs RunRegistry on the path; run this script "
            "from the repository root with PYTHONPATH=., as the usage line in this file's "
            "docstring does"
        ) from exc

    models = (
        ", ".join(args.models) if args.models else "every model in config.model_list"
    )
    scope = (
        f"{len(k_sample)} log-spaced k over {args.k_min:g}-{args.k_max:g} /Mpc, {models}, "
        f"{args.cpus} cpus; main.py args: {' '.join(main_args)}"
    )
    run = RunRegistry.begin(
        campaign=args.campaign,
        prompt=args.prompt,
        slug=args.register,
        purpose=args.purpose,
        script=__file__,
        # README §0.2: this job's checkpoint is its datastore, not a JSON-Lines file. The field
        # names the durable thing the results live in, which is what a resume needs; there is no
        # checkpoint.jsonl and record()/known() are never called on this run.
        checkpoint=str(database) if database is not None else None,
        scope=scope,
        heartbeat_means=HEARTBEAT_MEANS.format(interval=BEAT_MIN_INTERVAL),
    )

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
        f"**   list it with: PYTHONPATH=. ./venv/bin/python -m RunRegistry list"
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
        help="record this run in var/runs/ through RunRegistry, under this slug. Without it "
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
            raise RuntimeError(
                f"scoped_pipeline_run: datastore {db} already exists; refusing to reuse, "
                "overwrite or migrate it. Choose a path that does not exist."
            )
        if not db.parent.is_dir():
            raise RuntimeError(
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

    # (1) bootstrap Ray locally rather than attaching to a cluster
    import ray

    _real_ray_init = ray.init

    def _local_ray_init(*_args, **_kwargs):
        print(
            f"** scoped_pipeline_run: bootstrapping a local Ray instance "
            f"(num_cpus={args.cpus}, include_dashboard=False)"
        )
        return _real_ray_init(
            num_cpus=args.cpus, include_dashboard=False, ignore_reinit_error=True
        )

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
    if run is None:
        exec(compile(source, str(MAIN_PY), "exec"), namespace)
        return

    # the same exec, with a terminal state written after it. A SIGKILL never reaches here, which
    # is why the registry reports a run that still says `running` with a stale heartbeat rather
    # than inferring a death it did not see.
    try:
        exec(compile(source, str(MAIN_PY), "exec"), namespace)
    except SystemExit as exc:
        run.finish(terminal_state(exc.code), exit_code=exc.code)
        raise
    except BaseException:
        run.finish("failed", exit_code=1)
        # and re-raise, with the copies still open: the traceback the interpreter is about to
        # print is the one thing a failed run's directory most needs, and it is printed after
        # this frame unwinds. The kernel closes the copies when the process goes.
        raise
    run.finish("done", exit_code=0)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, RegisteredStream):
            stream.close_copy()


if __name__ == "__main__":
    main()
