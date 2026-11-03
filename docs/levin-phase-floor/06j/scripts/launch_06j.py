"""levin-phase-floor prompt 06j: write a run's plan and launch run_06j.py on it as one registered, detached run
(campaign levin-phase-floor, prompt 06j). New in 06j, after 06g's launch_refs.py (state/scripts_06g_reference/):
``runregistrykit.begin`` first (manifest, script run_06j.py), then ``nohup sh -c '<driver>; exit $?'`` in a new
session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid recorded with
``Run.heartbeat(pid=...)``; the driver replaces it with its own.

Plans:
- ``run1``: the captures (x1_capture.py on out/sample/selection.json, six workers), their check (check_captures.py),
  slicing (slice_06j.py), Gate J0's replays (replay_06j.py, every sliced call, six at once; s00 a second time in
  a fresh process for the determinism clause), the tool's re-read of 06g's r51 QAWO r1 (one_reading.py, six
  workers), Part S's probes (probe_s.py, every sliced S call, six at once) and Part W's preparation (w_prep.py).

- ``readings`` (state/readings_plan_20261103T0935Z.md, checked by the orchestrator): the 68 reference readings
  (QAWO and FCC at resolutions 1 and 2 on f00-f04 and c00-c11), one at a time, each by one_reading.py with six
  workers and --progress-order unordered, in ascending order of envelope projection
  (out/cost_projection_v2.json), each with the driver's monitor (fit coefficients a, b; envelope).

Usage (from var/levin-phase-floor-06j/): python -B scripts/launch_06j.py run1|readings
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
import refs_common as RC  # noqa: E402

PYTHON = str(RC.PYTHON)
TREE = str(RC.G_TREE)
SCRIPTS = str(HERE)
SCRIPT = HERE / "run_06j.py"
R51 = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/references/state/inputs/r51.pkl"


def py(script, *args):
    return [PYTHON, "-B", str(HERE / script), *[str(a) for a in args]]


def plan_run1():
    sel = json.loads((ROOT / "out" / "sample" / "selection.json").read_text())
    sids = [e["sid"] for e in sel]
    s_sids = [e["sid"] for e in sel if e["part"] == "S"]
    tp = f"{TREE}:{SCRIPTS}"
    stages = [
        {"name": "capture", "max_parallel": 1, "stop_on_fail": False, "jobs": [
            {"name": "30_capture", "pythonpath": TREE,
             "argv": py("x1_capture.py", "out/sample/selection.json", "state/capture", "out/capture.json", "--workers", 6)}]},
        {"name": "check", "max_parallel": 1, "jobs": [
            {"name": "31_check_captures", "pythonpath": tp, "requires": "out/capture.json",
             "argv": py("check_captures.py", "out/capture.json", "out/sample/selection.json", "out/check_captures.json")}]},
        {"name": "slice", "max_parallel": 1, "jobs": [
            {"name": "32_slice", "pythonpath": tp, "requires": "out/check_captures.json",
             "argv": py("slice_06j.py", "out/capture.json", "out/check_captures.json", "out/sample/selection.json")}]},
        {"name": "replays", "max_parallel": 6, "jobs": [
            {"name": f"33_replay_{s}", "pythonpath": tp, "requires": f"state/inputs/{s}.pkl",
             "argv": py("replay_06j.py", s, f"out/replay/{s}.json")} for s in sids]
         + [{"name": "34_replay_s00_second", "pythonpath": tp, "requires": "state/inputs/s00.pkl",
             "argv": py("replay_06j.py", "s00", "out/replay_det/s00-second.json")}]},
        {"name": "tool", "max_parallel": 1, "jobs": [
            {"name": "35_tool_reread_r51_qawo_r1", "pythonpath": tp,
             "argv": py("one_reading.py", R51, "qawo", 1, "out/j0_tool/r51-qawo-r1.json", "--sid", "r51", "--unit",
                        "g-1e-8-1-14231", "--index", 1, "--rtol", "1e-8", "--workers", 6)}]},
        {"name": "partS", "max_parallel": 6, "jobs": [
            {"name": f"36_probe_{s}", "pythonpath": tp, "requires": f"state/inputs/{s}.pkl",
             "argv": py("probe_s.py", s, f"out/partS/{s}.json.gz")} for s in s_sids]},
        {"name": "wprep", "max_parallel": 1, "jobs": [
            {"name": "37_w_prep", "pythonpath": tp, "argv": py("w_prep.py")}]},
    ]
    return {"label": "run1", "stages": stages}


def plan_readings():
    sel = {e["sid"]: e for e in json.loads((ROOT / "out" / "sample" / "selection.json").read_text())}
    cp = json.loads((ROOT / "out" / "cost_projection_v2.json").read_text())
    sids = ["f00", "f01", "f02", "f03", "f04"] + [f"c{i:02d}" for i in range(12)]
    rows = [p for p in cp["projection"] if p["sid"] in sids]
    rows.sort(key=lambda p: (p["projected_s"], p["sid"], p["impl"], p["res"]))
    tp = f"{TREE}:{SCRIPTS}"
    jobs = []
    for k, p in enumerate(rows):
        e = sel[p["sid"]]
        f = cp["fits"][f"{p['impl']}-r{p['res']}"]
        name = f"{60 + k:03d}_{p['sid']}-{p['impl']}-r{p['res']}"
        prog = f"out/progress/{p['sid']}-{p['impl']}-r{p['res']}.json"
        jobs.append({"name": name, "pythonpath": tp, "requires": f"state/inputs/{p['sid']}.pkl",
                     "argv": py("one_reading.py", ROOT / "state" / "inputs" / f"{p['sid']}.pkl", p["impl"], p["res"],
                                f"out/readings/{p['sid']}-{p['impl']}-r{p['res']}.json", "--sid", p["sid"], "--unit", e["unit"],
                                "--index", e["index"], "--rtol", e["rtol"], "--workers", 6, "--progress", prog,
                                "--progress-order", "unordered"),
                     "monitor": {"sid": p["sid"], "progress": prog, "a": f["s_per_segment"], "b": f["us_per_rad"] * 1e-6,
                                 "envelope_s": p["projected_s"], "fit_s": p["fit_s"]}})
    (ROOT / "out" / "progress").mkdir(parents=True, exist_ok=True)
    return {"label": "readings", "stages": [{"name": "readings", "max_parallel": 1, "jobs": jobs}]}


PURPOSES = {
    "run1": ("levin-phase-floor prompt 06j on the Mac Studio (mps026298; Python 3.13.16): captures, Gate J0, Part S and "
             "Part W's preparation. At the worktree /Users/ds283/Documents/Code/SecondaryGWKit-lpf06j-dfa19d9 (dispatch "
             "SHA dfa19d98; code 1c5336e's, tool b5626ecb's): 46 calls (Part S's 24, W-5's 10, W-12's 12) captured by "
             "06c's x1_capture.py (06g's copy) with 06g's census harness, reading 06's certified scratch copy of A3 v2 "
             "read-only (fingerprint 0e32f24a) on a stand-in ShardedPool, each checked bit for bit against 06g's census; "
             "J0's replays (every capture replayed alone; s00 twice) and the tool's re-read of 06g's r51 QAWO r1; Part S's "
             "probes of the leaf rule in isolation (S-a to S-e); Part W's preparation (phase spans, W-5's rtol equality, "
             "06h's leaf values, the constant move). No reference reading. At most six computing processes; Ray-free, no "
             "lock, no store built, no code changed"),
    "readings": ("levin-phase-floor prompt 06j on the Mac Studio (mps026298; Python 3.13.16): Part W's reference readings, per "
                 "state/readings_plan_20261103T0935Z.md (checked by the orchestrator). tools/filon_reference.py at the worktree "
                 "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06j-dfa19d9 (dfa19d98; tool b5626ecb's), QAWO and FCC at resolutions 1 "
                 "and 2 on 17 captured Route B calls (W-5's rows f00-f04, whose 1e-8 twins are equal, and W-12's c00-c11), 68 readings "
                 "one at a time with six workers each, cheapest first; projected 2.55 h (fit) to 6.06 h (envelope). Backstop: a "
                 "reading past 30 min or past its envelope is reported every 10 min; it is stopped only if its projection counting "
                 "every chunk in flight as finished is a day or more. Ray-free, no lock, no store built, no code changed"),
}
SCOPES = {
    "run1": ("out/sample/selection.json (46 calls); outputs under var/levin-phase-floor-06j/: state/capture, state/inputs, "
             "out/capture.json, out/check_captures.json, out/inputs.json, out/replay, out/replay_det, out/j0_tool, out/partS, "
             "out/w_prep, out/cmd/run1"),
    "readings": ("68 readings of state/inputs/{f00..f04,c00..c11}.pkl; outputs under var/levin-phase-floor-06j/: out/readings, "
                 "out/progress, out/reading_monitor.log, out/cmd/readings, out/readings_results.json"),
}
HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if label not in PURPOSES:
        raise SystemExit(f"usage: launch_06j.py {'|'.join(PURPOSES)}")
    plan = {"run1": plan_run1, "readings": plan_readings}[label]()
    plan_p = ROOT / "state" / f"plan_{label}.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists; not overwritten")
    plan_p.write_text(json.dumps(plan, indent=1))
    n = sum(len(st["jobs"]) for st in plan["stages"])
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="06j", slug=label, purpose=PURPOSES[label], argv=argv, cwd=str(ROOT),
        script=str(SCRIPT), unit="commands", expected_units=n, checkpoint=True, results=None, scope=SCOPES[label],
        heartbeat_means=HEARTBEAT_MEANS)
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
    sys.exit(0)
