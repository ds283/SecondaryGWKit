"""levin-phase-floor prompt 07a Part A: launch_07a.py for the attribution: run_attr_registered_07a.py on a list of
calls, as a registered detached run. launch_07a.py's text follows.

levin-phase-floor prompt 07a Part A: launch run_registered_07a.py on a plan as a registered detached run (CLAUDE.md,
"Long-running jobs -- the run registry": manifest first, nohup, output into the run directory, pid recorded). New in
07a, after 07's launch_census_07.py.

Usage: PYTHONPATH=<tree> python -B launch_07a.py SLUG PLAN_JSON OUT_DIR LOG_DIR DONE_JSON PURPOSE [--procs 6] [--timeout S]
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd"
SCRIPT = HERE / "run_attr_registered_07a.py"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("purpose")
    ap.add_argument("workers", type=int)
    ap.add_argument("sids", nargs="+")
    a = ap.parse_args()
    plan = a.sids
    a.procs = a.workers
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="07a", slug=a.slug,
        purpose=(a.purpose + f" (Mac Studio; tree {TREE} at f93aafd0, code 07's; script {SCRIPT.name} "
                 f"sha256 {sha256(SCRIPT)[:12]}; attr_07a.py sha256 {sha256(HERE / 'attr_07a.py')[:12]}; {len(plan)} calls: {' '.join(plan)}; "
                 f"at most {a.procs} processes; Ray-free, no store, no lock)"),
        argv=[PY, "-B", str(SCRIPT), "<run dir>", str(a.workers)] + plan, cwd=str(ROOT), script=str(SCRIPT),
        unit="calls", expected_units=len(plan), checkpoint=False, results=None,
        scope=f"Filon leaf references (tools.filon_reference, d37e633b) of captured calls (records in {ROOT}/out/attr); nothing written in the tree",
        heartbeat_means="at start with the driver's pid, after every job, and every 60 s by a thread of the driver")
    env = dict(os.environ)
    env["PYTHONPATH"] = TREE + ":" + str(HERE)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    command = " ".join(shlex.quote(x) for x in [PY, "-B", str(SCRIPT), run.path, str(a.workers)] + plan) + "; exit $?"
    child = subprocess.Popen(["nohup", "sh", "-c", command], cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    stdout.close()
    stderr.close()
    run.heartbeat(pid=child.pid, stage="launched (detached sh started)")
    print(f"run id {run.id}\nrun dir {run.path}\nsh pid {child.pid}\ncommand {command}")


if __name__ == "__main__":
    main()
    sys.exit(0)
