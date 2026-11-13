"""levin-phase-floor prompt 06g section 5, the references: launch run_refs.py as one registered, detached
run (campaign levin-phase-floor, prompt 06g, slug ``references``). Copied from 06h's phase agent's
launch_06h.py (kept in state/scripts_06h_phase_reference/) and changed only in its script, slug,
purpose, scope and unit count: ``runregistrykit.begin`` first (manifest, script run_refs.py), then
``nohup sh -c '<driver>; exit $?'`` in a new session, stdin from /dev/null, stdout and stderr into the
run directory, the sh pid recorded with ``Run.heartbeat(pid=...)``; the driver replaces it with its
own. New in 06g.
Usage (from var/levin-phase-floor-06g/references/): python -B scripts/launch_refs.py 1|2  (the part)
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_refs import expected_units, plan  # noqa: E402

PYTHON = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
SCRIPT = HERE / "run_refs.py"
SELECTION = json.loads((ROOT / "out" / "sample" / "selection.json").read_text())
I06, NEW77, V8 = plan(SELECTION)
PART = sys.argv[1] if len(sys.argv) > 1 else None
if PART not in ("1", "2"):
    raise SystemExit("usage: launch_refs.py 1|2")
PURPOSE_ALL = (
    "levin-phase-floor prompt 06g section 5 on the Mac Studio (mps026298; Python 3.13.16): the references "
    "(V3, V4, the correction -sum E). At the code worktree /Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-1c5336e "
    "(Part B's dispatch SHA 1c5336e, code 06g's 3ae6f3a): captures Z3's 30 calls and 60 calls drawn from 06h's census "
    "by 06c's x1_capture.py (06g's copy) with the g census harness, each checked bit for bit (value, abserr, label, "
    "converged, regions) against the g census checkpoint, reading 06's certified scratch copy of A3 v2 read-only "
    "(fingerprint 0e32f24a) on a stand-in ShardedPool; 06i's 13 groups compared with 06i's inputs; then "
    "tools/filon_reference.py (QAWO and FCC, two resolutions each, six worker processes) on 06i's 13 captures, "
    "compared with 06i's readings (a disagreement stops the run), and on the 77 new samples, each reading in its own "
    "process group, stopped at 30 minutes. At most six computing processes; Ray-free, no lock, no store built, no code changed"
)
PURPOSE = (
    f"[part {PART} of 2: "
    + (
        "captures (Z3's 30, 60 drawn, addendum 2's V8 calls), their checks, 06i's 13 groups, the tool on 06i's 13 and the comparison with 06i's readings; part 2 is launched after the cost projection] "
        if PART == "1"
        else "out/part2_plan.json's readings, per addendum 3 (the user's ruling of 2026-11-01): addendum 1's V1 12 (captured at g against R24, bit for bit), addendum 2's V8 12, the rest of V3's 77 (FCC only on the 17 large-span Route A calls), then QAWO res 1 on ten of those; no reading stopped at 30 minutes: past max(30 min, its projection) its progress and projected finish are logged, and the backstop stops it only at a projected total of a day, at max(6 x projection, projection + 1 h), or at a day] "
    )
    + PURPOSE_ALL
)
SCOPE = (
    f"out/sample/selection.json ({len(SELECTION)} calls: Z3's 30, 13 of them 06i's, and 60 drawn); outputs under "
    "var/levin-phase-floor-06g/references/ (state/capture, state/inputs, out/capture.json, out/check_captures.json, "
    "out/inputs.json, out/compare13.json, out/routes.json, out/readings_06i13, out/cli, out/tool_vs_06i.json, "
    "out/readings, out/reading_logs, out/cmd/30-35)"
)
HEARTBEAT_MEANS = (
    "Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
    "stage's and each reading's start and end, and every 60 s by a thread of the driver while its process lives."
)


def main():
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--part", PART]
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="06g",
        slug="references",
        purpose=PURPOSE,
        argv=argv,
        cwd=str(ROOT),
        script=str(SCRIPT),
        unit="units",
        expected_units=expected_units(PART, len(I06), len(NEW77), len(V8)),
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
            for a in [PYTHON, "-B", str(SCRIPT), "--run-dir", run.path, "--part", PART]
        )
        + "; exit $?"
    )
    child = subprocess.Popen(
        ["nohup", "sh", "-c", command],
        cwd=str(ROOT),
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
    print(f"script      {SCRIPT}")
    print(f"sh pid      {child.pid}")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
