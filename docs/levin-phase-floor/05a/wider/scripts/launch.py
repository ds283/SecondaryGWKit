"""
levin-phase-floor prompt 05a section 6 (copied from prompt 05's scripts/launch.py,
docs/levin-phase-floor/05/wider/scripts/launch.py, sha256 d8ac0dc3..., a record that is not
edited). Changed here, each change marked "prompt 05a": the prompt (``05a``) under which the run is
registered, the trees and the unit sets named in its purpose and scope, and the names it imports
from prov.py. Prompt 05's text follows, unchanged; read its trees ``p``/``c`` as ``c``/``t``, its
``53084dc`` as ``149c393`` and its ``e648951`` as ``c4eaefd``.

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

from prov import C_SHA, C_TREE, MAIN, PYTHON, T_SHA, WIDER, sha256  # noqa: E402  prompt 05a

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_wider.py"

PURPOSE = (  # prompt 05a
    "levin-phase-floor prompt 05a section 6 (T1 as amended by section 9) on the Mac Studio "
    "(mps026298; Python 3.13.16): the wider re-measurement of T's node-anchored remainder on A3 v2, "
    "Ray-free, no lock. Two trees: c, the code commit's parent "
    f"{C_SHA[:7]} (05's c) in the git worktree {C_TREE}; t, the code commit 1fc349c, in the main "
    f"checkout at {T_SHA[:7]} (this manifest's git_head; its code is 1fc349c's up to bd38e50's "
    "comments). 05's 39 rows of state/sample.json (31 depth-20, 8 controls), re-run in-process on "
    "a read-only stand-in ShardedPool with QuadSource recomputed in-process; units in order "
    "c@1e-7 (39), t@1e-7 (39), t@1e-8 (31 depth-20); t budget 3x the row's c wall; the phase probe "
    "in every 1e-7 unit, after its timed section; about 20-60 min"
)
SCOPE = (  # prompt 05a
    "39 QuadSourceIntegral rows of handover-A3-v2-lambdacdm.sqlite (read-only), listed in "
    "var/levin-phase-floor-05a/wider/state/sample.json (05's sample, copied unchanged); trees c "
    "(worktree c4eaefd) and t (main checkout, code 1fc349c); rtol 1e-7 and 1e-8 (t, depth-20 "
    "rows); no store write, no Ray, no lock"
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
        prompt="05a",  # prompt 05a
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
