"""levin-phase-floor prompt 07c section 5 (Part C), the R32 rehearsal agent: write a plan and launch 07b Part A's plan driver
run_06j.py (07b's rehearsal copy, unchanged, scripts/run_06j.py) on it as one registered, detached run (campaign
levin-phase-floor, prompt 07c). 07b's rehearsal launch_r31.py (98890b5e) with its r31-refs plan removed and the r31-replay
plan carried over as r32-replay: ``runregistrykit.begin`` first (manifest, script run_06j.py), then
``nohup sh -c '<driver>; exit $?'`` in a new session, stdin from /dev/null, stdout and stderr into the run directory, the sh
pid recorded with ``Run.heartbeat(pid=...)``; the driver replaces it with its own.

Plan ``r32-replay`` (slug ``r32-replay``): the replay of R32's located rows, P1's, the w rows, the block rows and the row of
the record of interest, at the tree h (cc3368ce, code 67b5af74), by the capture harness copied from 07b's rehearsal
(scripts/capture/, changed where marked "prompt 07c r32"), then its check against R32, the four-block check's pass 2 and
check, and the capture of the group record of interest and its check. Each stage one command, ``stop_on_fail``. The
harness runs at most six worker processes at once. Ray-free; under the held rehearsal lock (no Ray is started); no store
written; nothing written in a tree.

Usage (any cwd): python -B launch_r32.py r32-replay
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # var/levin-phase-floor-07c/rehearsal
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
PYTHON = str(MAIN / "venv" / "bin" / "python")
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c"
SCRIPT = HERE / "run_06j.py"
RP = HERE / "capture" / "replay07c_r32.py"
ST = ROOT / "state"


def rp(*args):
    return [PYTHON, "-B", str(RP), *[str(a) for a in args]]


def plan_replay():
    stages = [
        ("replay_run", rp("run", ST / "replay_spec.json")),
        ("replay_check", rp("check", ST / "capture" / "replay", ST / "replay_rows.json", ROOT / "out" / "replay_calls.csv")),
        ("blocks2_spec", rp("spec", "blocks2", ST / "replay_rows.json", ST / "blocks2_spec.json")),
        ("blocks2_run", rp("run", ST / "blocks2_spec.json")),
        ("blockcheck", rp("blockcheck", ST / "replay_rows.json", ROOT / "out" / "blocks.json")),
        ("refcap_spec", rp("refcapspec", ST / "replay_rows.json", ST / "refcap_spec.json")),
        ("refcap_run", rp("run", ST / "refcap_spec.json")),
        ("refcapcheck", rp("refcapcheck", ST / "replay_rows.json", ROOT / "out" / "refcap.json")),
    ]
    return {"label": "r32-replay", "stages": [
        {"name": n, "max_parallel": 1, "stop_on_fail": True,
         "jobs": [{"name": f"{i:02d}_{n}", "pythonpath": TREE, "argv": argv}]}
        for i, (n, argv) in enumerate(stages, 1)]}


PURPOSES = {
    "r32-replay": (
        "levin-phase-floor prompt 07c section 5 (Part C), the R32 rehearsal agent, on the Mac Studio (mps026298; Python "
        "3.13.16): the replay behind S1, S2 and S4, the chase-stop counts and the call of section 9.5's S3 record, as 07b's "
        "rehearsal replayed R31. 07b's rehearsal capture harness (scripts/capture/, changed where marked 'prompt 07c r32': "
        f"tree h = the worktree {TREE} at cc3368ce, code 67b5af74; R32's store read-only through state/stores/r32; each "
        "call's num_chase_stops, num_chase_stopped_leaves and num_chase_noise_bisections recorded beside its refusals) "
        "re-runs 203 R32 rows (202 located, P1, the w rows, the two block rows, the row of the 1 record of interest: "
        "converged changed) and checks every record bit for bit against R32; then the four S4 blocks' pass 2 and check "
        "(each capture against R31's), and the capture of the group record of interest, checked against R32. Ray-free "
        "(never initialised), at most six worker processes, under the held rehearsal lock, no store written, nothing "
        "written in a tree"),
}
SCOPES = {
    "r32-replay": ("state/replay_spec.json (203 units), state/blocks2_spec.json, state/refcap_spec.json; outputs under "
                   "var/levin-phase-floor-07c/rehearsal/: state/capture/{replay,blocks2,refcap}, state/inputs/, "
                   "out/replay_calls.csv, out/replay_calls.checks.json, out/blocks.json, out/refcap.json, out/cmd/r32-replay"),
}
HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if label not in PURPOSES:
        raise SystemExit(f"usage: launch_r32.py {'|'.join(PURPOSES)}")
    plan = {"r32-replay": plan_replay}[label]()
    plan_p = ROOT / "state" / f"plan_{label}.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists; not overwritten")
    plan_p.write_text(json.dumps(plan, indent=1))
    n = sum(len(st["jobs"]) for st in plan["stages"])
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="07c", slug=label, purpose=PURPOSES[label], argv=argv, cwd=str(ROOT),
        script=str(SCRIPT), unit="commands", expected_units=n, checkpoint=True, results=None,
        scope=SCOPES[label], heartbeat_means=HEARTBEAT_MEANS)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    command = " ".join(shlex.quote(a) for a in [PYTHON, "-B", str(SCRIPT), "--run-dir", run.path, "--plan", str(plan_p)]) + "; exit $?"
    child = subprocess.Popen(["nohup", "sh", "-c", command], cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    stdout.close()
    stderr.close()
    run.heartbeat(pid=child.pid, stage="launched (detached sh started)")
    print(f"run id      {run.id}")
    print(f"run dir     {run.path}")
    print(f"plan        {plan_p} ({n} commands)")
    print(f"sh pid      {child.pid}")
    print(f"command     {command}")
    print(f"launcher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
