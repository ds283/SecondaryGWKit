"""Send one signal (SIGTERM, or SIGKILL with `--signal KILL`) to a registered run at a chosen stage.

Written for prompt 10 of prompts/datastore-integrity (§1.4), which interrupts a scoped build
twice and resumes it, to show that the resume recomputes nothing it had stored. It is a one-shot:
it signals at most once, and then it exits.

Usage, launched detached right after the build it watches (CLAUDE.md, "Long-running jobs"):

    nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \\
        var/runs/<run id> '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, ' 40 \\
        [--signal {TERM,KILL}] > /dev/null 2>&1 &
    disown

Every `--interval` seconds (15) it reads the run's `status.json`, and:

- if the run is in a terminal state, or its pid no longer answers `kill -0`, it logs that and exits
  **without signalling**;
- while `runregistrykit.liveness(status, --stale-after)` is `stale` with the pid still alive, it waits
  and never signals: a long work item makes the heartbeat go stale, and stale means "go and look";
- if the run is `alive` and its `stage` matches the regex (`re.search`) with a progress reading at
  or above the percentage, it checks that the pid still answers `kill -0` and that
  `ps -p <pid> -o command=` names the run's own driver and `--register <slug>`. The driver is the
  basename of the manifest's `argv[0]`, the script the registry recorded as launched
  (`scoped_pipeline_run.py` for a scoped pipeline run, `quadsource_atol_sweep.py` for a `--build`),
  and the slug is the one after `--register` in the same `argv`. Only then does it send the signal
  once, log it and exit. If the command line does not name the run, it logs that and exits without
  signalling. A manifest with no `argv[0]`, like one with no slug, makes it log that and exit
  without signalling.

`--signal {TERM,KILL}` (default `TERM`) chooses the signal. A `KILL` runs no handler in the run, so
its status stays `running` and its end is never recorded: it is ended afterwards with
`python -m runregistrykit end RUN_ID --reason TEXT` (CLAUDE.md, "Long-running jobs", rule 5). Every
check before the signal is the same for both, and so is the log, bar the signal's name.

The percentage is the one `scoped_pipeline_run.StageTracker` writes into `stage`,
`"<title>, <p>% (<n> of <m> work items remaining)"`. A stage that has printed only its banner reads
as 0%, so a percentage of 0 interrupts at the stage's first line. A stage reading
`"<title>, all work items complete"` is never eligible: the stage is over, and signalling then
would interrupt between stages, which is not what was asked.

`--hold S` (default 0) is for a stage that prints no intermediate progress line, which can then only
be interrupted from its banner: after the first eligible reading it waits `S` seconds, reads
`status.json` again, and signals only if that reading is eligible too, so that the stage has had
time to store some of its rows. Otherwise it carries on watching.

It never identifies a process by a command-line pattern (CLAUDE.md rule 2): the pid comes from
`status.json`, and `ps` only confirms that the pid is still the run's.

Its log is `<run dir>/interrupter.log`, appended to, one timestamped line per event, beginning with
its own pid and arguments; a traceback goes there too.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import runregistrykit  # noqa: E402

_PERCENT = re.compile(r",\s*([\d.]+)%\s*\(")
_COMPLETE = re.compile(r",\s*all work items complete\s*$")


def now() -> str:
    return runregistrykit.now_iso()


class Log:
    def __init__(self, path):
        self._handle = open(path, "a", buffering=1)

    def __call__(self, text):
        self._handle.write(f"{now()} {text}\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    @property
    def handle(self):
        return self._handle


def run_slug(run_dir: Path):
    """The `--register` slug from the run's manifest `argv`, or `None`."""
    manifest = runregistrykit.read_json(str(run_dir / "manifest.json")) or {}
    argv = manifest.get("argv") or []
    for i, token in enumerate(argv[:-1]):
        if token == "--register":
            return argv[i + 1]
    return None


def run_driver(run_dir: Path):
    """The basename of the run's own driver, from its manifest's `argv[0]`, or `None`.

    `runregistrykit.begin` records `sys.argv` before a driver replaces it, so `argv[0]` is the script
    that was launched: `scoped_pipeline_run.py`, or `quadsource_atol_sweep.py` for a `--build`.
    """
    manifest = runregistrykit.read_json(str(run_dir / "manifest.json")) or {}
    argv = manifest.get("argv") or []
    if not argv or not isinstance(argv[0], str):
        return None
    return os.path.basename(argv[0]) or None


def progress(stage):
    """The stage's percentage: its reading, 0 for a bare banner, `None` once it is complete."""
    if stage is None:
        return None
    if _COMPLETE.search(stage):
        return None
    reading = _PERCENT.search(stage)
    if reading is not None:
        return float(reading.group(1))
    return 0.0


def command_of(pid):
    """`ps -p <pid> -o command=`, or `None` if `ps` finds nothing."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stdout.strip()
    return text or None


def names_the_run(command, slug, driver) -> bool:
    if command is None or slug is None or driver is None:
        return False
    tokens = command.split()
    named = any(
        i + 1 < len(tokens) and tokens[i] == "--register" and tokens[i + 1] == slug
        for i in range(len(tokens))
    )
    return driver in command and named


def watch(
    run_dir: Path,
    pattern,
    minimum,
    interval,
    stale_after,
    log,
    hold=0.0,
    sig=signal.SIGTERM,
) -> int:
    stage_re = re.compile(pattern)
    slug = run_slug(run_dir)
    driver = run_driver(run_dir)
    log(
        f"interrupter pid {os.getpid()} watching {run_dir} (slug {slug!r}); stage regex "
        f"{pattern!r} at >= {minimum}%; every {interval:g} s; stale after {stale_after:g} s; "
        f"hold {hold:g} s"
    )
    if slug is None:
        log("the run's manifest names no --register slug; exiting without signalling")
        return 2
    if driver is None:
        log("the run's manifest names no argv[0]; exiting without signalling")
        return 2
    last = None
    held = hold <= 0
    while True:
        status = runregistrykit.read_json(str(run_dir / "status.json")) or {}
        state = status.get("state")
        pid = status.get("pid")
        stage = status.get("stage")
        if state in runregistrykit.TERMINAL_STATES:
            log(
                f"run is {state!r} (exit code {status.get('exit_code')!r}); exiting without signalling"
            )
            return 0
        if state == "running" and not runregistrykit.pid_alive(pid):
            log(
                f"run's pid {pid} no longer answers kill -0; exiting without signalling"
            )
            return 0
        verdict = runregistrykit.liveness(status, stale_after)
        reading = (verdict, stage)
        if reading != last:
            log(f"liveness {verdict}; pid {pid}; stage {stage!r}")
            last = reading
        if verdict == "alive" and stage is not None and stage_re.search(stage):
            percent = progress(stage)
            if percent is not None and percent >= minimum and not held:
                log(
                    f"eligible at stage {stage!r}; holding {hold:g} s, then reading again"
                )
                held = True
                time.sleep(hold)
                continue
            if percent is not None and percent >= minimum:
                if not runregistrykit.pid_alive(pid):
                    log(
                        f"pid {pid} no longer answers kill -0; exiting without signalling"
                    )
                    return 0
                command = command_of(pid)
                if not names_the_run(command, slug, driver):
                    log(
                        f"pid {pid}'s command line does not name {driver} (the run's own "
                        f"driver, its manifest's argv[0]) and --register {slug}: "
                        f"{command!r}; exiting without signalling"
                    )
                    return 2
                os.kill(int(pid), sig)
                log(
                    f"sent {signal.Signals(sig).name} to pid {pid} at stage {stage!r} ({percent:g}% >= {minimum:g}%); "
                    f"command {command!r}; exiting"
                )
                return 0
        time.sleep(interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "run_dir", type=Path, help="the run's directory under var/runs/"
    )
    parser.add_argument("stage", help="a regex, searched for in status.json's stage")
    parser.add_argument(
        "percent", type=float, help="the minimum progress reading, in %%"
    )
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument(
        "--hold",
        type=float,
        default=0.0,
        help="seconds to wait after the first eligible reading, before reading again and "
        "signalling only if that reading is eligible too (default 0)",
    )
    parser.add_argument(
        "--signal",
        choices=["TERM", "KILL"],
        default="TERM",
        help="the signal to send (default TERM). KILL runs no handler: the run's end is then "
        "recorded with `python -m runregistrykit end`",
    )
    parser.add_argument(
        "--stale-after", type=float, default=runregistrykit.DEFAULT_STALE_AFTER
    )
    parser.add_argument(
        "--log", type=Path, default=None, help="default <run dir>/interrupter.log"
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    log = Log(args.log or run_dir / "interrupter.log")
    sys.stderr = log.handle
    return watch(
        run_dir,
        args.stage,
        args.percent,
        args.interval,
        args.stale_after,
        log,
        hold=args.hold,
        sig=getattr(signal, "SIG" + args.signal),
    )


if __name__ == "__main__":
    sys.exit(main())
