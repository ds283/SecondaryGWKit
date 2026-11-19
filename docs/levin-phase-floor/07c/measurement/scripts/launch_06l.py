"""levin-phase-floor prompt 06l: write a run's plan and launch 06j's driver run_06j.py (copied unchanged) on it
as one registered, detached run (campaign levin-phase-floor, prompt 06l). New in 06l, copied from 06k's
launch_06k.py (a record; its copy is in state/scripts_06k_reference/), with its plans replaced: the launch itself
(``runregistrykit.begin`` first, manifest, script run_06j.py, then ``nohup sh -c '<driver>; exit $?'`` in a new
session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid recorded with
``Run.heartbeat(pid=...)``; the driver replaces it with its own) is 06k's, unchanged.

Plans:
- ``k0partL``: Part L's captures (06j's x1_capture.py on 06k's out/sample/selection.json, six workers), their check
  against 06g's census (check_captures.py) and slicing (slice_06j.py); Gate K0 (run_k0.py on 06k's out/k0/units.json
  under the identity variant and route B, one after the other, six workers each; compare_k0.py on each: identity
  mode for the identity variant, route mode for route B); Part L's probes (probe_l.py on every sliced call, six at
  once); the LiouvilleGreen draws (calibration/lg_seeded_draws.py run, six processes, modes tree, identity and
  routeB).

- ``partL``: added after run levin-phase-floor-06l-k0partL-20261105T091942, whose 31_check_captures failed every
  capture on the harness check alone (run_census.py, which 06k changed after its captures and which the check did not
  allow) and so sliced none and ran no probe: the check again (check_captures.py, now allowing run_census.py) on that
  run's out/capture.json into out/check_captures_b.json, the slicing into state/inputs/ and out/inputs_b.json, and
  Part L's probes (probe_l.py on every sliced call, six at once). The captures, K0 and the draws are not re-run.

Usage (from var/levin-phase-floor-06l/): python -B scripts/launch_06l.py k0partL|partL
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
VARIANTS = ("identity", "routeB")


def py(script, *args):
    return [PYTHON, "-B", str(HERE / script), *[str(a) for a in args]]


def plan_k0partL():
    sel = json.loads((ROOT / "out" / "sample" / "selection.json").read_text())
    sids = [e["sid"] for e in sel]
    tp = f"{TREE}:{SCRIPTS}"
    stages = [
        {"name": "capture", "max_parallel": 1, "jobs": [
            {"name": "30_capture", "pythonpath": TREE,
             "argv": py("x1_capture.py", "out/sample/selection.json", "state/capture", "out/capture.json", "--workers", 6)}]},
        {"name": "check", "max_parallel": 1, "jobs": [
            {"name": "31_check_captures", "pythonpath": tp, "requires": "out/capture.json",
             "argv": py("check_captures.py", "out/capture.json", "out/sample/selection.json", "out/check_captures.json")}]},
        {"name": "slice", "max_parallel": 1, "jobs": [
            {"name": "32_slice", "pythonpath": tp, "requires": "out/check_captures.json",
             "argv": py("slice_06j.py", "out/capture.json", "out/check_captures.json", "out/sample/selection.json")}]},
    ]
    for k, v in enumerate(VARIANTS):
        stages.append({"name": f"k0_{v}", "max_parallel": 1, "jobs": [
            {"name": f"{41 + k}_k0_run_{v}", "pythonpath": TREE,
             "argv": py("run_k0.py", v, "out/k0/units.json", f"state/k0/{v}", "--workers", 6)}]})
    stages.append({"name": "k0_compare", "max_parallel": 2, "jobs": [
        {"name": f"{51 + k}_k0_compare_{v}", "pythonpath": tp, "requires": f"state/k0/{v}/units.jsonl",
         "argv": py("compare_k0.py", "identity" if v == "identity" else "route", f"state/k0/{v}",
                    f"out/k0/compare_{v}.json")} for k, v in enumerate(VARIANTS)]})
    stages.append({"name": "partL", "max_parallel": 6, "jobs": [
        {"name": f"60_probe_l_{s}", "pythonpath": tp, "requires": f"state/inputs/{s}.pkl",
         "argv": py("probe_l.py", s, f"out/partL/{s}.json")} for s in sids]})
    stages.append({"name": "lg_draws", "max_parallel": 1, "jobs": [
        {"name": "70_lg_draws", "pythonpath": TREE,
         "argv": py("calibration/lg_seeded_draws.py", "run", "--processes", 6)}]})
    return {"label": "k0partL", "stages": stages}


def plan_partL():
    """prompt 06l, after run k0partL's check failed on the harness check alone: check, slice, probes."""
    sel = json.loads((ROOT / "out" / "sample" / "selection.json").read_text())
    sids = [e["sid"] for e in sel]
    tp = f"{TREE}:{SCRIPTS}"
    return {"label": "partL", "stages": [
        {"name": "check", "max_parallel": 1, "jobs": [
            {"name": "31b_check_captures", "pythonpath": tp, "requires": "out/capture.json",
             "argv": py("check_captures.py", "out/capture.json", "out/sample/selection.json", "out/check_captures_b.json")}]},
        {"name": "slice", "max_parallel": 1, "jobs": [
            {"name": "32b_slice", "pythonpath": tp, "requires": "out/check_captures_b.json",
             "argv": py("slice_06j.py", "out/capture.json", "out/check_captures_b.json", "out/sample/selection.json",
                        ROOT / "out" / "inputs_b.json")}]},
        {"name": "partL", "max_parallel": 6, "jobs": [
            {"name": f"60_probe_l_{s}", "pythonpath": tp, "requires": f"state/inputs/{s}.pkl",
             "argv": py("probe_l.py", s, f"out/partL/{s}.json")} for s in sids]}]}


PURPOSES = {
    "k0partL": ("levin-phase-floor prompt 06l on the Mac Studio (mps026298; Python 3.13.16): Gate K0 and Part L, before "
                "planned stop 1. At the worktree /Users/ds283/Documents/Code/SecondaryGWKit-lpf06l-af4c4dc (dispatch SHA "
                "af4c4dc1; production code 1c5336e's, 06g's): 06j's 46 calls re-captured with 06g's census harness, reading "
                "06's certified scratch copy of A3 v2 read-only (fingerprint 0e32f24a) on a stand-in ShardedPool, each checked "
                "bit for bit against 06g's census; Gate K0, 06k's 100 census units run under the identity variant and route B "
                "(scratch variants of AdaptiveLevin/levin_quadrature.py under var/levin-phase-floor-06l/scripts/variants/, "
                "loaded in place of the tree's module), each compared with 06g's census records; Part L's leaf-level probes "
                "of route B on 06j's 142 slow leaves, the 06h carrying leaves of the 22 W calls and the 24 S calls' trusted "
                "leaves; 06g's 20 seeded LiouvilleGreen draws under the tree, the identity variant and route B. No census. At "
                "most six computing processes; Ray-free, no lock, no store built, no code changed in the tree"),
    "partL": ("levin-phase-floor prompt 06l on the Mac Studio (mps026298; Python 3.13.16): Part L's check, slicing and "
              "probes again, after run levin-phase-floor-06l-k0partL-20261105T091942's capture check failed every capture "
              "on its harness-script check alone (run_census.py not allowed; the captured values matched 06g's census). "
              "check_captures.py (now allowing run_census.py) on that run's 46 captures, the slicing, and probe_l.py on "
              "every sliced call (route B's leaf-level numbers beside 06k's), at the worktree "
              "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06l-af4c4dc (af4c4dc1). No census, no store read but 06g's "
              "census records; at most six computing processes; Ray-free, no lock, no code changed in the tree"),
}
SCOPES = {
    "k0partL": ("out/sample/selection.json (46 calls), out/k0/units.json (100 units); outputs under "
                "var/levin-phase-floor-06l/: state/capture, state/inputs, out/capture.json, out/check_captures.json, "
                "out/inputs.json, state/k0/<variant>, out/k0/compare_<variant>.json, out/partL, out/lg_draws, "
                "out/cmd/k0partL"),
    "partL": ("out/capture.json and state/capture (run k0partL's, read only); outputs under var/levin-phase-floor-06l/: "
              "out/check_captures_b.json, state/inputs, out/inputs_b.json, out/partL, out/cmd/partL"),
}

HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if label not in PURPOSES:
        raise SystemExit(f"usage: launch_06l.py {'|'.join(PURPOSES)}")
    plan = {"k0partL": plan_k0partL, "partL": plan_partL}[label]()  # prompt 06l
    plan_p = ROOT / "state" / f"plan_{label}.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists; not overwritten")
    plan_p.write_text(json.dumps(plan, indent=1))
    n = sum(len(st["jobs"]) for st in plan["stages"])
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="06l", slug=label, purpose=PURPOSES[label], argv=argv, cwd=str(ROOT),
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
