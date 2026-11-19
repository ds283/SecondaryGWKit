"""levin-phase-floor prompt 07c Part A: 07b Part A's scripts/launch_07b.py (a record) copied, changed only where marked
"prompt 07c": the prompt recorded in the registry is "07c", and ROOT is this records directory (var/levin-phase-floor-07c/
measurement), the driver its scripts/run_06j.py (byte-identical to 07b's). 07b Part A's text follows.

levin-phase-floor prompt 07b Part A: launch one plan as a registered, detached run (runregistrykit; campaign
levin-phase-floor, prompt 07b), driven by 07a's run_06j.py (copied unchanged), as 07a's launch_07a.py did. New in 07b,
after launch_07a.py (a record). The plan is a JSON file already written in state/ (by plan_07b.py), named on the command
line with its slug, purpose and scope; the launcher refuses a plan file it did not find, and a slug with a run already
recorded under that plan. The driver runs every job with cwd var/levin-phase-floor-07b/measurement (run_06j.py's ROOT)
and the job's own PYTHONPATH, each script by its absolute path (no ``python -m``).

Usage: python -B launch_07b.py SLUG PLAN_JSON PURPOSE_TXT SCOPE_TXT
"""

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
SCRIPT = HERE / "run_06j.py"
HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    slug, plan_p, purpose_p, scope_p = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]), Path(sys.argv[4])
    plan = json.loads(plan_p.read_text())
    if plan.get("label") != slug:
        raise SystemExit("the plan's label is not the slug")
    n = sum(len(st["jobs"]) for st in plan["stages"])
    purpose = purpose_p.read_text().strip()
    scope = scope_p.read_text().strip()
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="07c", slug=slug, purpose=purpose, argv=argv, cwd=str(ROOT),
        script=str(SCRIPT), unit="commands", expected_units=n, checkpoint=True, results=None, scope=scope,
        heartbeat_means=HEARTBEAT_MEANS)  # prompt 07c: prompt "07c"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    command = " ".join(shlex.quote(a) for a in [PYTHON, "-B", str(SCRIPT), "--run-dir", run.path, "--plan", str(plan_p)]) + "; exit $?"
    child = subprocess.Popen(["nohup", "sh", "-c", command], cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    stdout.close()
    stderr.close()
    run.heartbeat(pid=child.pid, stage="launched (detached sh started)")
    print(f"run id      {run.id}")
    print(f"run dir     {run.path}")
    print(f"plan        {plan_p} ({n} commands)")
    print(f"sh pid      {child.pid}")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
