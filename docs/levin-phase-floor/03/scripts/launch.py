"""
levin-phase-floor prompt 03, step 6: launch the registered run, detached.

1. ``runregistrykit.begin`` writes the manifest and status (campaign levin-phase-floor, prompt 03,
   slug instrument) before anything runs; its ledger is ``checkpoint.jsonl`` in the run directory.
2. ``run_instrument.py --run-dir RUN_DIR`` is started under ``nohup`` in a new session
   (``start_new_session=True``; ``setsid`` the command is not on macOS), stdin from /dev/null,
   stdout and stderr appended to the run directory's ``stdout.log`` and ``stderr.log``, with
   ``OMP_NUM_THREADS=1`` (what Ray sets for a worker with the default one CPU, as
   ``compute_QuadSource_integral`` ran under) and ``PYTHONPATH`` the repository root.
3. The detached child's pid is recorded with ``Run.heartbeat(pid=...)``.

The launcher then exits; it does not wait on the child.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_instrument.py"
PYTHON = REPO / "venv" / "bin" / "python"

PURPOSE = (
    "levin-phase-floor prompt 03 on the Mac Studio (mps026298; tree 98e68d3; Python 3.13.16): "
    "re-run the seven depth-20 A3 v2 QuadSourceIntegral rows R1-R7 of prompt section 1 "
    "in-process, Ray-free on a read-only stand-in ShardedPool, with transparent wrappers on "
    "adaptive_levin_sincos and build_phase_groups; M0 bit-identity per row; per-row pickles in "
    "this run directory; about 25-30 min"
)
SCOPE = (
    "7 rows of handover-A3-v2-lambdacdm.sqlite (read-only): (2,37527) (0,1797) (1,37670) "
    "(2,37720) (0,37966) (2,34728) (2,34820); no store write, no Ray"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the detached pid, by the child once at start, and at "
    "each row's start and end (a row runs 2-6 min with no heartbeat in between). A row's ledger "
    "record means its M0 comparison was made and its pickle written."
)


def main():
    child_argv = [str(PYTHON), str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="03",
        slug="instrument",
        purpose=PURPOSE,
        argv=child_argv,
        cwd=str(REPO),
        script=str(SCRIPT),
        unit="rows",
        expected_units=7,
        checkpoint=True,
        results=None,
        scope=SCOPE,
        heartbeat_means=HEARTBEAT_MEANS,
    )
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(REPO)
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    child = subprocess.Popen(
        ["nohup", str(PYTHON), str(SCRIPT), "--run-dir", run.path],
        cwd=str(REPO),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        close_fds=True,
    )
    stdout.close()
    stderr.close()
    run.heartbeat(pid=child.pid, stage="launched (detached child started)")
    print(f"run id      {run.id}")
    print(f"run dir     {run.path}")
    print(f"manifest    {run.manifest_path}")
    print(f"child pid   {child.pid} (nohup execs python in the same process)")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
