"""levin-phase-floor prompt 06g section 5, the references: launch run_rerun.py as one registered,
detached run (campaign levin-phase-floor, prompt 06g, slug ``references-rerun``), per addenda 4 and
5. Copied from launch_refs.py (itself 06h's launch_06h.py) and changed only in its script, slug,
purpose, scope and unit count. New in 06g.
Usage (from var/levin-phase-floor-06g/references/): python -B scripts/launch_rerun.py"""

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
SCRIPT = HERE / "run_rerun.py"
PLAN = json.loads((ROOT / "out" / "rerun_plan.json").read_text())
PURPOSE = (
    "levin-phase-floor prompt 06g section 5 on the Mac Studio (mps026298; Python 3.13.16): the references' rerun batch "
    "(addenda 4 and 5, the user's rulings of 2026-11-02), every reading by the Filon tool at ff3e8394 (worktree "
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-ff3e839: Route B's stationary-point segments integrated directly), "
    "one at a time with six workers, progress counted out of order: six bit-for-bit re-reads of clean part 1 and 2 readings "
    "(a difference stops the run), all four readings of the 14 stationary-point samples (w00-w11, z00, z12), then the "
    "stopped QAWO readings that run (r24 r1, r53 r1, r47 r2, r53 r2, r26 r1); a reading is stopped only at a projected "
    "total of a day or 24 h run. At most six computing processes; Ray-free, no lock, no store opened, no code changed"
)
SCOPE = (
    f"out/rerun_plan.json ({PLAN['n_readings']} readings; estimate {PLAN['estimate_total_s'][0]:.0f}-"
    f"{PLAN['estimate_total_s'][1]:.0f} s); outputs under var/levin-phase-floor-06g/references/ (out/rereads, "
    "out/readings/*-rerun.json, out/reading_logs/rerun, out/progress_rerun, out/rerun_monitor.log)"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each reading's start "
    "and end, at each 5-minute check past 30 minutes, and every 60 s by a thread of the driver while its process lives."
)


def main():
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="06g",
        slug="references-rerun",
        purpose=PURPOSE,
        argv=argv,
        cwd=str(ROOT),
        script=str(SCRIPT),
        unit="units",
        expected_units=PLAN["n_readings"],
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
