"""Launch run_fcc.py as one registered, detached run (campaign levin-phase-floor, prompt 06g, slug
``fcc-laptop``): ``runregistrykit.begin`` first, then ``nohup sh -c '<driver>; exit $?'`` in a new
session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid recorded with
``Run.heartbeat(pid=...)``; the driver replaces it with its own. After Part B's launch_refs.py.
Usage (from var/levin-phase-floor-06g-fcc-laptop/): python -B scripts/launch_fcc.py
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_fcc import TREE, units  # noqa: E402

PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
SCRIPT = HERE / "run_fcc.py"
PURPOSE = (
    "levin-phase-floor 06g references, on the laptop by the user's word (2026-11-02, the Mac Studio busy with "
    "the rerun batch's stage 3): FCC r1 and r2 re-read with tools/filon_reference.py at b5626ec (FCC integrates "
    "directly the segments beside a stationary point) on the 14 stationary-point samples w00-w11, z00, z12, "
    "from captures copied from the Mac Studio (SHA-256 checked); first four bit-for-bit checks against the "
    "Mac Studio's readings (w06 qawo r1, r55 fcc r1/r2, r55 qawo r1). Six worker processes; Ray-free, no lock, "
    f"no store. Tool from the worktree {TREE}"
)


def main():
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="06g",
        slug="fcc-laptop",
        purpose=PURPOSE,
        argv=argv,
        cwd=str(ROOT),
        script=str(SCRIPT),
        unit="readings",
        expected_units=len(units()),
        checkpoint=True,
        results=None,
        scope="var/levin-phase-floor-06g-fcc-laptop/ (inputs, readings)",
        heartbeat_means="Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each reading's start, and every 60 s by a thread of the driver.",
    )
    d = str(run.path)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TREE}:{ROOT / 'harness'}"
    driver = " ".join(shlex.quote(x) for x in [PYTHON, "-B", str(SCRIPT), "--run-dir", d])
    p = subprocess.Popen(
        ["nohup", "sh", "-c", f"{driver}; exit $?"],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=open(os.path.join(d, "stdout.log"), "ab"),
        stderr=open(os.path.join(d, "stderr.log"), "ab"),
        start_new_session=True,
    )
    run.heartbeat(pid=p.pid, stage="launched")
    print(d, p.pid)


if __name__ == "__main__":
    main()
