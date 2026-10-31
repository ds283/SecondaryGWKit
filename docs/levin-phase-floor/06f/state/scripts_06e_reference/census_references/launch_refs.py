"""levin-phase-floor prompt 06e section 5, Z3: launch samples 3 and 15's references as one registered,
detached run (campaign levin-phase-floor, prompt 06e, slug ``references``), as the census's
launch.py launches its run: ``runregistrykit.begin`` first (manifest, script run_refs.py), then
``nohup sh -c '<driver>; exit $?'`` in a new session, stdout and stderr into the run directory,
the sh pid recorded with ``Run.heartbeat(pid=...)``; the driver replaces it with its own. New in
06e. Usage (from references/): python -B launch_refs.py"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
SCRIPT = HERE / "run_refs.py"
PURPOSE = (
    "levin-phase-floor prompt 06e section 5, Z3 on the Mac Studio: Gauss-Legendre references for "
    "depth20-tiling's samples 3 (g-1e-7-3-34370 call 4, G-q-r) and 15 (g-1e-8-2-1549 call 4, G-q-r), "
    "by 06c's orchestrator's orch_ref.py (copied unchanged, sha256 8a0d291d) on depth20-tiling's "
    "captured inputs, at P=128 with 64/96 nodes and P=64 with 96/128 nodes, six workers, one "
    "invocation at a time, at the tree g (worktree 6d81fad); Ray-free, no lock, no store"
)
SCOPE = "4 invocations: s03-P128, s03-P64, s15-P128, s15-P64; outputs in var/levin-phase-floor-06e/census/references/out/"
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
    "invocation's start and end, and every 60 s by a thread of the driver while its process lives."
)


def main():
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(campaign="levin-phase-floor", prompt="06e", slug="references", purpose=PURPOSE,
                               argv=argv, cwd=str(HERE), script=str(SCRIPT), unit="invocations", expected_units=4,
                               checkpoint=True, results=None, scope=SCOPE, heartbeat_means=HEARTBEAT_MEANS)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    command = " ".join(shlex.quote(a) for a in [PYTHON, "-B", str(SCRIPT), "--run-dir", run.path]) + "; exit $?"
    child = subprocess.Popen(["nohup", "sh", "-c", command], cwd=str(HERE), env=env, stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
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
