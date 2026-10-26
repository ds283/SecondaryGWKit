"""
levin-phase-floor prompt 05 section 6: launch the registered run, detached. Adapted from prompt
04's ``scripts/launch.py`` (docs/levin-phase-floor/04/scripts/launch.py, sha256 51572fa8..., a
record that is not edited).

1. ``runregistrykit.begin`` writes the manifest and status (campaign levin-phase-floor, prompt 05,
   slug wider) before anything runs, with ``script=`` run_wider.py's path in the main checkout, so
   that the registry stamps its SHA-256 on every checkpoint record. The manifest's git_head is the
   main checkout's, ``53084dc``, which is the c tree; the p tree is the worktree at ``e648951``,
   which the purpose text names and every p checkpoint payload records (04's I4).
2. The driver runs under ``nohup sh -c '<driver>; exit $?'``: two commands, so that ``sh`` does
   not exec the driver but stays its parent, and reaps it if it is killed (CLAUDE.md, run registry
   rule 5).
   It starts in a new session (``start_new_session=True``, since ``setsid`` is not on macOS), stdin
   from /dev/null, stdout and stderr appended to the run directory's ``stdout.log`` and
   ``stderr.log``, with ``-B``, ``PYTHONDONTWRITEBYTECODE=1`` and no ``PYTHONPATH`` (the driver
   imports nothing from either tree; each worker is given its own), and its working directory
   ``var/levin-phase-floor-05/wider``.
3. The ``sh`` pid is recorded with ``Run.heartbeat(pid=...)`` at once; the driver replaces it with
   its own pid, the Python process, as its first act.

The launcher then exits; it does not wait on the child.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import C_SHA, MAIN, P_SHA, P_TREE, PYTHON, WIDER, sha256  # noqa: E402

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_wider.py"

PURPOSE = (
    "levin-phase-floor prompt 05 section 6 on the Mac Studio (mps026298; Python 3.13.16): the wider "
    "re-measurement of layer 1(a) on A3 v2, Ray-free, no lock. Two trees: p, the code commit's "
    f"parent {P_SHA[:7]} in the git worktree {P_TREE}; c, the code commit 120fdfe, in the main "
    f"checkout at {C_SHA[:7]} (this manifest's git_head; it adds only records). 39 rows of "
    "state/sample.json (31 depth-20, 8 controls), re-run in-process on a read-only stand-in "
    "ShardedPool with QuadSource recomputed in-process; units in order p@1e-7 (39), c@1e-7 (39), "
    "c@1e-8 (31 depth-20); c budget 3x the row's p wall; about 30-60 min"
)
SCOPE = (
    "39 QuadSourceIntegral rows of handover-A3-v2-lambdacdm.sqlite (read-only), listed in "
    "var/levin-phase-floor-05/wider/state/sample.json; trees p (worktree e648951) and c (main "
    "checkout, code 120fdfe); rtol 1e-7 and 1e-8 (c, depth-20 rows); no store write, no Ray, no "
    "lock"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at "
    "each worker's start and each unit's record, and every 60 s by a thread of the driver while "
    "its process lives. A unit's ledger record means it finished, done, over budget or in error, "
    "and its region pickle was written."
)


def main():
    sample = __import__("json").loads((WIDER / "state" / "sample.json").read_text())
    n_rows = len(sample["rows"])
    n_d20 = sum(1 for r in sample["rows"] if r["role"] == "d20")
    expected = 2 * n_rows + n_d20
    child_argv = [str(PYTHON), "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="05",
        slug="wider",
        purpose=PURPOSE,
        argv=child_argv,
        cwd=str(WIDER),
        script=str(SCRIPT),
        unit="units",
        expected_units=expected,
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
            shlex.quote(a)
            for a in [str(PYTHON), "-B", str(SCRIPT), "--run-dir", run.path]
        )
        + "; exit $?"
    )
    child = subprocess.Popen(
        ["nohup", "sh", "-c", command],
        cwd=str(WIDER),
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
    print(f"manifest    {run.manifest_path}")
    print(f"expected    {expected} units ({n_rows} rows x 2 at 1e-7, {n_d20} at 1e-8)")
    print(f"script      {SCRIPT} sha256 {sha256(SCRIPT)}")
    print(f"sh pid      {child.pid} (nohup execs sh; sh runs the driver as its child)")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
