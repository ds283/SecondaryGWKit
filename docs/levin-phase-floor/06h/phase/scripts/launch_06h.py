"""levin-phase-floor prompt 06h section 5, the phase re-measurement: launch run_06h.py as one registered,
detached run (campaign levin-phase-floor, prompt 06h, slug ``phase``). Copied from 06f's
launch_06f.py and changed only in its script, slug, purpose, scope and unit count:
``runregistrykit.begin`` first (manifest, script run_06h.py), then ``nohup sh -c '<driver>; exit
$?'`` in a new session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid
recorded with ``Run.heartbeat(pid=...)``; the driver replaces it with its own. New in 06h.
Usage (from var/levin-phase-floor-06h/phase/): python -B scripts/launch_06h.py"""

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
SCRIPT = HERE / "run_06h.py"
N_CALLS = len(json.loads((ROOT / "out" / "sample" / "selection_h.json").read_text()))
assert N_CALLS == len(
    json.loads((ROOT / "out" / "sample" / "selection_p.json").read_text())
)
PURPOSE = (
    "levin-phase-floor prompt 06h section 5 on the Mac Studio (mps026298; Python 3.13.16): the phase "
    "re-measurement, 06f's M2 items 1 and 4 repeated at 06h's code (H1, H2). Captures 06f's 20 M2 calls "
    "by 06e's x1_capture.py (06h's copy) at the parent p (worktree "
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-2c325f5, 2c325f5, code 06e's; each capture checked "
    "bit for bit against 06e's census checkpoint a0dc4d7f) and at h (worktree "
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-c1ec3b2, dispatch SHA c1ec3b2, code 7172a8f + "
    "ac1c6f4; each checked against 06h's census checkpoint), reading 06's certified scratch copy of A3 v2 "
    "read-only (fingerprint 0e32f24a) on a stand-in ShardedPool; the method check at the exact e^u before "
    "any scoring; replays each call in its own process in its own tree with 06f's pass-throughs, with "
    "mpmath phase references at 30 digits at the exact e^u (and at 06f's x_hat), at every node of every "
    "returned leaf and 06f's seeded dense sample of 1,000 points per call; each tree's code scored at the "
    "other's points; the reference re-checked at 50 digits at h's worst point. At most six processes; "
    "Ray-free, no lock, no store built, no code changed"
)
SCOPE = (
    f"out/sample/selection_p.json and selection_h.json ({N_CALLS} calls each, 06f's M2 sample, seed "
    "20261031); outputs under var/levin-phase-floor-06h/phase/ (state/capture_p, state/capture_h, "
    "out/capture_p.json, out/capture_h.json, out/replay_p, out/replay_h, out/replay_logs, out/cross, "
    "out/cross_logs, out/cmd/30-33)"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
    "stage's and each child's start and end, and every 60 s by a thread of the driver while its process lives."
)


def main():
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="06h",
        slug="phase",
        purpose=PURPOSE,
        argv=argv,
        cwd=str(ROOT),
        script=str(SCRIPT),
        unit="units",
        expected_units=2 + 1 + 2 * N_CALLS + 2 * N_CALLS + 1,
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
