"""prompt 07b extension (section 2.7): Part A's scripts/launch_07b.py copied, changed only in this note, ROOT (two levels
up) and its driver (scripts/ext/run_ext.py). Its text follows.

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
ROOT = HERE.parent.parent  # prompt 07b extension
PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
SCRIPT = HERE / "run_ext.py"  # prompt 07b extension
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
        campaign="levin-phase-floor", prompt="07b", slug=slug, purpose=purpose, argv=argv, cwd=str(ROOT),
        script=str(SCRIPT), unit="commands", expected_units=n, checkpoint=True, results=None, scope=scope,
        heartbeat_means=HEARTBEAT_MEANS)
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
