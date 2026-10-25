"""
levin-phase-floor prompt 04, step 7: launch the registered run, detached (prompt 03's launch.py,
adapted to prompt 04 and to prompt section 8's provenance).

1. ``runregistrykit.begin`` writes the manifest and status (campaign levin-phase-floor, prompt 04,
   slug remove) before anything runs, with ``script=`` run_remove.py's path in the main checkout,
   so that the registry stamps its SHA-256 on every checkpoint record. The manifest's git_head is
   therefore the main checkout's (the records, c8ca155); the code is the worktree's 98e68d3, which
   the purpose text names and every checkpoint payload records (prompt section 8).
2. ``run_remove.py --run-dir RUN_DIR`` is started under ``nohup`` in a new session
   (``start_new_session=True``), stdin from /dev/null, stdout and stderr appended to the run
   directory's ``stdout.log`` and ``stderr.log``, with ``PYTHONPATH`` the worktree,
   ``PYTHONDONTWRITEBYTECODE=1`` and ``-B`` (nothing is written in the worktree),
   ``OMP_NUM_THREADS=1`` (as 03), and its working directory ``var/levin-phase-floor-04``.
3. The detached child's pid is recorded with ``Run.heartbeat(pid=...)``.

The launcher then exits; it does not wait on the child.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import MAIN, WORKTREE, WORKTREE_SHA, sha256  # noqa: E402

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_remove.py"
PYTHON = MAIN / "venv" / "bin" / "python"
CWD = HERE.parent

PURPOSE = (
    "levin-phase-floor prompt 04 on the Mac Studio (mps026298; Python 3.13.16): remove the floor. "
    f"The code runs from the git worktree {WORKTREE} at {WORKTREE_SHA} (98e68d3, the tree prompt 03 "
    "ran at; this manifest's git_head is the main checkout's, where the scripts live). Re-runs the "
    "seven depth-20 A3 v2 QuadSourceIntegral rows R1-R7 of prompt 03 in-process, Ray-free on a "
    "read-only stand-in ShardedPool, under the arms of prompt 04 section 2 patched at "
    "PrimitivePhase.theta_mod_2pi: 28 units in order P@1e-7, X@1e-7, A@1e-7, X@1e-8 x R1..R7; "
    "M0 on P; X/A budget 3x the row's P wall; up to about 4 h"
)
SCOPE = (
    "7 rows of handover-A3-v2-lambdacdm.sqlite (read-only): (2,37527) (0,1797) (1,37670) "
    "(2,37720) (0,37966) (2,34728) (2,34820); arms P, X, A; rtol 1e-7 and 1e-8 (X); no store "
    "write, no Ray, no lock"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the detached pid, by the child at start, at each unit's "
    "start and end, and every 60 s by a thread of the child while its process lives (a unit can "
    "run longer than the staleness window). A unit's ledger record means it finished, within "
    "budget or over it, and (for X and A) its pickle was written."
)


def main():
    child_argv = [str(PYTHON), "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="04",
        slug="remove",
        purpose=PURPOSE,
        argv=child_argv,
        cwd=str(CWD),
        script=str(SCRIPT),
        unit="units",
        expected_units=28,
        checkpoint=True,
        results=None,
        scope=SCOPE,
        heartbeat_means=HEARTBEAT_MEANS,
    )
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(WORKTREE)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    child = subprocess.Popen(
        ["nohup", str(PYTHON), "-B", str(SCRIPT), "--run-dir", run.path],
        cwd=str(CWD),
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
    print(f"script      {SCRIPT} sha256 {sha256(SCRIPT)}")
    print(f"child pid   {child.pid} (nohup execs python in the same process; no sh)")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
