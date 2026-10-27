"""
handover prompt 13 (B2), Part B: launch the registered run, detached. Adapted from A4's
`a4_launch.py` (a record that is not edited).

    ./venv/bin/python -B var/handover-13/scripts/b2_launch.py    (from the worktree's root)

1. `runregistrykit.begin` writes the manifest and status (campaign handover, prompt 13, slug
   residue) before anything runs, with `script=` the driver `b2_driver.py`, so that the registry
   stamps its SHA-256 on every checkpoint record. A run launched from a git worktree is recorded
   in the main checkout's `var/runs/`.
2. The driver runs under `nohup sh -c '<driver>; exit $?'`: two commands, so that `sh` does not
   exec the driver but stays its parent, and reaps it if it is killed (CLAUDE.md, run registry
   rule 5). It starts in a new session (`start_new_session=True`; `setsid` is not on macOS),
   stdin from /dev/null, stdout and stderr appended to the run directory's `stdout.log` and
   `stderr.log`, with `-B`, `PYTHONDONTWRITEBYTECODE=1` and no `PYTHONPATH` (`b2_cells` finds the
   repository itself), and its working directory `var/handover-13`.
3. The `sh` pid is recorded with `Run.heartbeat(pid=...)` at once; the driver replaces it with its
   own pid, the Python process, as its first act.

The launcher then exits; it does not wait on the child. No Ray, no datastore, no lock. After
landing, this file is a record of its own SHA-256.
"""

import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKDIR = HERE.parent
DRIVER = HERE / "b2_driver.py"
PYTHON = Path(sys.executable)

import runregistrykit  # noqa: E402

PURPOSE = (
    "handover prompt 13 (B2), Part B, on the laptop (Python 3.13.16) in the worktree "
    "SecondaryGWKit-handover-13 at 28ade78 (Part A landed): score the residue the T_k seam gap "
    "was masking. A2's ladder (together 980-1e8, T-first 980-1e7, q-smooth 980-1e7; b = 0) in 13 "
    "objects built through test_quadsource_integral.Case: E (exact flavour), R (production's "
    "construction and seam), E' (R's geometry, exact ingredients) and the add-one and remove-one "
    "hybrids of five terms (f the source spline, Tnum the numeric T, Trep T's stored WKB "
    "representation M and phase, TLG the Liouville-Green omega and dlnM_dz, G the Green's "
    "function's phase and amplitude). 208 cells, each a child process, at most 2 at once, 1 h "
    "wall-clock budget per cell (over budget = not reached); Ray-free, no datastore, no lock; "
    "expect under an hour"
)
SCOPE = (
    "208 cells = 16 rungs x 13 objects of b2_cells.py (imports docs/handover/realistic_large_x.py "
    "c1cd3598, unedited); REF_ATOL 1e-45, REF_RTOL 1e-12; no store, no Ray, no lock; records in "
    "this run directory only"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at "
    "each cell's record, and every 60 s by a thread of the driver while its process lives. A "
    "cell's checkpoint record means it ended: ok, not reached (killed over its 1 h budget), raised "
    "or skipped."
)


def sha256(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    child_argv = [str(PYTHON), "-B", str(DRIVER), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="handover",
        prompt="13",
        slug="residue",
        purpose=PURPOSE,
        argv=child_argv,
        cwd=str(WORKDIR),
        script=str(DRIVER),
        unit="cells",
        expected_units=208,
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
            for a in [str(PYTHON), "-B", str(DRIVER), "--run-dir", run.path]
        )
        + "; exit $?"
    )
    child = subprocess.Popen(
        ["nohup", "sh", "-c", command],
        cwd=str(WORKDIR),
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
    print(f"checkpoint  {run.checkpoint_path}")
    print(f"expected    208 cells")
    print(f"driver      {DRIVER} sha256 {sha256(DRIVER)}")
    print(f"sh pid      {child.pid} (nohup execs sh; sh runs the driver as its child)")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
