"""
levin-phase-floor prompt 06 section 6 (copied from prompt 05a's scripts/launch.py,
docs/levin-phase-floor/05a/wider/scripts/launch.py, a record that is not edited). Changed here,
each change marked "prompt 06": the prompt (``06``) under which the run is registered, the trees
and unit sets named in its purpose and scope, and the names it imports from prov.py. The
manifest's git_head is the main checkout's (``28ebaf0``), where no tree runs; both trees are
worktrees, which the purpose names and every checkpoint payload records. 05a's text follows,
unchanged; read its ``c``/``t`` as ``t5``/``s``.

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

from prov import MAIN, PYTHON, S_SHA, S_TREE, T5_SHA, T5_TREE, WIDER, sha256  # noqa: E402  prompt 06

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_wider.py"

PURPOSE = (  # prompt 06
    "levin-phase-floor prompt 06 section 6 on the Mac Studio (mps026298; Python 3.13.16): the "
    "re-measurement of 05's 39 rows on A3 v2 with the producers re-run in-process, Ray-free, no "
    "lock. Two trees, both git worktrees: t5, the code commit's parent "
    f"{T5_SHA[:7]} (05a's t; its consumers and the stored producer samples) in {T5_TREE}; s, "
    f"Part B's dispatch SHA {S_SHA[:7]} (code dc9d28c; its producers re-run on each row's "
    f"GkWKBIntegration and TkWKBIntegration objects, and its consumers) in {S_TREE}. The manifest's "
    "git_head is the main checkout's (28ebaf0), where no tree runs. 05's 39 rows of "
    "state/sample.json (31 depth-20, 8 controls), on a read-only stand-in ShardedPool with QuadSource "
    "recomputed in-process; units in order t5@1e-7 (39), s@1e-7 (39), s@1e-8 (31 depth-20); s budget "
    "3x the row's t5 wall (the producer re-run is outside it, in the rebuild); the phase probe in the "
    "1e-7 units of R1-R7, after the timed section"
)
SCOPE = (  # prompt 06
    "39 QuadSourceIntegral rows of handover-A3-v2-lambdacdm.sqlite (read-only), listed in "
    "var/levin-phase-floor-06/wider/state/sample.json (05's sample, copied unchanged); trees t5 "
    "(worktree 9f66622) and s (worktree f5cee4c, code dc9d28c); rtol 1e-7 and 1e-8 (s, depth-20 "
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
        prompt="06",  # prompt 06
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
