"""
handover prompt 04 (A4): launch the registered run, detached. Adapted from
`docs/levin-phase-floor/05/wider/scripts/launch.py` (a record that is not edited).

    ./venv/bin/python -B var/handover-04/scripts/a4_launch.py      (from the main checkout's root)

1. `runregistrykit.begin` writes the manifest and status (campaign handover, prompt 04, slug
   cost-growth) before anything runs, with `script=` the driver `a4_driver.py`, so that the
   registry stamps its SHA-256 on every checkpoint record. The client root is found by walking up
   from the script's directory to `runregistrykit.toml`.
2. The driver runs under `nohup sh -c '<driver>; exit $?'`: two commands, so that `sh` does not
   exec the driver but stays its parent, and reaps it if it is killed (CLAUDE.md, run registry
   rule 5). It starts in a new session (`start_new_session=True`; `setsid` is not on macOS),
   stdin from /dev/null, stdout and stderr appended to the run directory's `stdout.log` and
   `stderr.log`, with `-B`, `PYTHONDONTWRITEBYTECODE=1` and no `PYTHONPATH` (`a4_cells` finds the
   repository itself), and its working directory `var/handover-04`.
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
DRIVER = HERE / "a4_driver.py"
PYTHON = Path(sys.executable)

import runregistrykit  # noqa: E402

PURPOSE = (
    "handover prompt 04 (A4) on the Mac Studio (mps026298; Python 3.13.16), main checkout at the "
    "dispatch SHA 28ebaf0: does A2's Levin cost growth survive production's phases? A2's ladder "
    "(together 980-1e8, T-first 980-1e7, q-smooth 980-1e6; b = 0; one seam) at HEAD in four arms "
    "that differ in G's phase alone: E exact; S A2's phase_spline G; P0 a PrimitivePhase G with "
    "exact_remainder off; P production's G (exact_remainder on); T production's (table cs_tau, "
    "keyword at its default, production seam) in S, P0, P. 60 cells, each a child process, up to "
    "4 at once, 3 h wall-clock budget per cell (over budget = not reached); Ray-free, no "
    "datastore, no lock; expect a few hours"
)
SCOPE = (
    "60 cells = 15 rungs x arms E, S, P0, P of a4_cells.py (imports docs/handover/"
    "realistic_large_x.py c1cd3598, unedited); REF_ATOL 1e-45, REF_RTOL 1e-12; no store, no Ray, "
    "no lock; records in this run directory only"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at "
    "each cell's record, and every 60 s by a thread of the driver while its process lives. A "
    "cell's checkpoint record means it ended: ok, not reached (killed over its 3 h budget), raised "
    "or skipped."
)


def sha256(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    child_argv = [str(PYTHON), "-B", str(DRIVER), "--run-dir", "<run dir>"]
    run = runregistrykit.begin(
        campaign="handover",
        prompt="04",
        slug="cost-growth",
        purpose=PURPOSE,
        argv=child_argv,
        cwd=str(WORKDIR),
        script=str(DRIVER),
        unit="cells",
        expected_units=60,
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
            shlex.quote(a) for a in [str(PYTHON), "-B", str(DRIVER), "--run-dir", run.path]
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
    print(f"expected    60 cells")
    print(f"driver      {DRIVER} sha256 {sha256(DRIVER)}")
    print(f"sh pid      {child.pid} (nohup execs sh; sh runs the driver as its child)")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
