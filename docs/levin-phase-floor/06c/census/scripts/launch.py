"""
levin-phase-floor prompt 06c section 5 (copied from prompt 06b's census scripts/launch.py,
docs/levin-phase-floor/06b/census/scripts/launch.py, a record that is not edited; its copy is kept
in state/scripts_06b_reference/). Changed here, each change marked "prompt 06c": the prompt
(``06c``) under which the run is registered (slug ``census``, as 06b's), the purpose and scope (the
tree g, its SHA and the two code commits', the copy's fingerprint, the row count, the two rtols,
06b's run as the before, what each unit records), and the names imported from prov.py. 06b's text
follows, unchanged; read its b as g.

levin-phase-floor prompt 06b section 5 (copied from prompt 06a's scripts/launch.py,
docs/levin-phase-floor/06a/scripts/launch.py, a record that is not edited; its copy is kept in
state/scripts_06a_reference/). Changed here, each change marked "prompt 06b": the prompt (``06b``)
under which the run is registered (slug ``census``, as 06a's), the purpose and scope (the tree b,
its SHA and the code commit's, the copy's fingerprint, the row count, the two rtols, W3's evidence
and W2's capture), and the names imported from prov.py. The manifest's git_head is the main
checkout's (``28ebaf0``), where no tree code runs; the tree is the worktree b, which the purpose
names and every checkpoint payload records. 06a's text follows, unchanged; read its h as b.

levin-phase-floor prompt 06a (copied from prompt 06's scripts/launch.py,
docs/levin-phase-floor/06/wider/scripts/launch.py, a record that is not edited). Changed here, each
change marked "prompt 06a": the prompt (``06a``) and slug (``census``) under which the run is
registered, the driver (run_census.py), the purpose and scope (prompt 06a section 6: the tree's
SHA, the copy's fingerprint, the row count and the two rtols), the expected units (from
state/rows.json) and the names imported from prov.py. The manifest's git_head is the main
checkout's (``28ebaf0``), where no tree code runs; the tree is the worktree h, which the purpose
names and every checkpoint payload records. 06's text follows, unchanged.

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

from prov import (  # noqa: E402  prompt 06c
    A3V2_DIGEST,
    CODE_SHA_06C,
    CODE_SHA_06D,
    G_SHA,
    G_TREE,
    P_SHA,
    PYTHON,
    RUN06B,
    WIDER,
    sha256,
)

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_census.py"  # prompt 06a

ROWS = WIDER / "state" / "rows.json"  # prompt 06a
N_ROWS = len(__import__("json").loads(ROWS.read_text())["rows"])
PURPOSE = (  # prompt 06c
    "levin-phase-floor prompt 06c section 5 on the Mac Studio (mps026298; Python 3.13.16): 06b's "
    f"census re-run under the global error budget, at 06c's code {CODE_SHA_06C} (one budget for the "
    f"call, the trust check one level down) with 06d's {CODE_SHA_06D} (the Clenshaw-Curtis round-off "
    f"floor), every A3 v2 QuadSourceIntegral row ({N_ROWS}) re-run in-process at rtol 1e-7 and 1e-8 "
    f"(atol 1e-32), Ray-free, no lock, against 06b's run {RUN06B.name} as the before. One tree, g, "
    f"the git worktree at Part B's dispatch SHA {G_SHA[:7]} ({G_TREE}; its code outside docs/ and "
    f"prompts/ is {CODE_SHA_06C}'s plus {CODE_SHA_06D}'s; the parent {P_SHA[:7]}, Gate X0 showed, "
    "reproduces 06b). It reads, read-only, 06's certified scratch copy of A3 v2 "
    f"(var/levin-phase-floor-06/wider/state/a3v2_copy, fingerprint {A3V2_DIGEST[:8]}) on a stand-in "
    "ShardedPool, with 06a's producer cache. One worker per k per rtol, at most six at once; every "
    "1e-7 unit before any 1e-8 unit; 300 s budget per unit; per call, the global budget's counters, "
    "the solves by purpose and every returned leaf's trust-check numbers; in every unit, after its "
    "timed section, the deepest line of each call that reaches depth_max; in every 1e-7 unit, 06a's "
    "stationary-point scan and 06b's recomputed floors of every phase-limited region. The "
    "manifest's git_head is the main checkout's (28ebaf0), where no tree code runs; 06b's run took "
    "51 min, and this one may take much longer"
)
SCOPE = (  # prompt 06c
    f"all {N_ROWS} QuadSourceIntegral rows of handover-A3-v2-lambdacdm (read-only, through 06's "
    "certified scratch copy), listed in var/levin-phase-floor-06c/census/state/rows.json (06a's, "
    f"sha256 aa523d28); tree g (worktree {G_SHA[:7]}, code {CODE_SHA_06C} + {CODE_SHA_06D}); rtol "
    "1e-7 and 1e-8; no store write, no Ray, no lock"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at "
    "each worker's start and each unit's record, and every 60 s by a thread of the driver while "
    "its process lives. A unit's ledger record means it finished, done, over budget or in error, "
    "and its region pickle was written."
)


def main():
    expected = 2 * N_ROWS  # prompt 06a: every row at 1e-7, then at 1e-8
    child_argv = [str(PYTHON), "-B", str(SCRIPT), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="06c",  # prompt 06c
        slug="census",
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
    print(f"expected    {expected} units ({N_ROWS} rows at 1e-7, then at 1e-8)")  # prompt 06a
    print(f"script      {SCRIPT} sha256 {sha256(SCRIPT)}")
    print(f"sh pid      {child.pid} (nohup execs sh; sh runs the driver as its child)")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
