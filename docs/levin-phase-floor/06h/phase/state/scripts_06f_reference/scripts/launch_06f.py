"""levin-phase-floor prompt 06f: launch run_06f.py as one registered, detached run (campaign
levin-phase-floor, prompt 06f, slug ``measure``), as 06e's launch_refs.py launches its run:
``runregistrykit.begin`` first (manifest, script run_06f.py), then ``nohup sh -c '<driver>; exit $?'``
in a new session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid recorded
with ``Run.heartbeat(pid=...)``; the driver replaces it with its own. New in 06f.
Usage (from var/levin-phase-floor-06f/): python -B scripts/launch_06f.py"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
SCRIPT = HERE / "run_06f.py"
N_SEL = len(json.loads((ROOT / "out" / "sample" / "selection.json").read_text()))
PURPOSE = (
    "levin-phase-floor prompt 06f on the Mac Studio (mps026298; Python 3.13.16): how two Clenshaw-Curtis "
    "floor terms add up across a call's leaves. Captures the 41 sampled calls of 06e's census "
    "(levin-phase-floor-06e-census-20261030T081318, checkpoint a0dc4d7f) by 06e's x1_capture.py at the "
    "tree g (worktree /Users/ds283/Documents/Code/SecondaryGWKit-lpf06f-f4770d1, dispatch SHA f4770d13, "
    "code f4d8e17's), reading 06's certified scratch copy of A3 v2 read-only (fingerprint 0e32f24a) on a "
    "stand-in ShardedPool; replays each in its own process with pass-throughs at _adaptive_levin_subregion, "
    "_cc_node_term_parts, _cc_roundoff_floor and _declared_endpoint_phase_err (M1: Z2's call and 20 "
    "node-dominated calls; M2: 20 declared-dominated analytic-block calls, with mpmath phase references at "
    "30 digits and Gauss-Legendre responses); Z2's call replayed twice; item 1's method check. At most six "
    "processes; Ray-free, no lock, no store built, no code changed"
)
SCOPE = (
    f"out/sample/selection.json ({N_SEL} calls, seed 20261031); outputs under var/levin-phase-floor-06f/"
    "(state/capture, out/capture.json, out/replay, out/replay_logs, out/cmd/30_capture.txt, "
    "out/cmd/31_method_check.txt)"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
    "stage's and each replay's start and end, and every 60 s by a thread of the driver while its process lives."
)


def main():
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="06f",
        slug="measure",
        purpose=PURPOSE,
        argv=argv,
        cwd=str(ROOT),
        script=str(SCRIPT),
        unit="units",
        expected_units=N_SEL + 3,
        checkpoint=True,
        results=None,
        scope=SCOPE,
        heartbeat_means=HEARTBEAT_MEANS,
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    command = (
        " ".join(
            shlex.quote(a) for a in [PYTHON, "-B", str(SCRIPT), "--run-dir", run.path]
        )
        + "; exit $?"
    )
    child = subprocess.Popen(
        ["nohup", "sh", "-c", command],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        close_fds=True,
    )
    stdout.close()
    stderr.close()
    run.heartbeat(pid=child.pid, stage="launched (detached sh started)")
    print(f"run id      {run.id}")
    print(f"run dir     {run.path}")
    print(f"script      {SCRIPT}")
    print(f"sh pid      {child.pid}")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
