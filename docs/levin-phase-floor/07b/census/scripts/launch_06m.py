"""levin-phase-floor prompt 06m section 5: write a run's plan and launch 06j's driver run_06j.py (copied unchanged) on it
as one registered, detached run (campaign levin-phase-floor, prompt 06m). Copied from 06l's launch_06l.py (a record,
not edited), with its plans replaced, each change marked "prompt 06m". The launch itself is 06l's (and 06k's),
unchanged: ``runregistrykit.begin`` first (manifest, script run_06j.py), then ``nohup sh -c '<driver>; exit $?'`` in a
new session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid recorded with
``Run.heartbeat(pid=...)``; the driver replaces it with its own.

Plans:
- ``k0`` (Gate M0, slug ``k0``): 06k's 100 K0 units (docs/levin-phase-floor/06l/out/k0/units.json in the tree) run by
  run_k0.py at the tree with no variant (VARIANT ``tree``: the tree's own AdaptiveLevin/levin_quadrature.py, 06m's
  code), six workers, into state/k0/tree; then compare_06m.py (mode k0) against route B's census into
  out/k0/compare_m0.json.

Usage (from var/levin-phase-floor-06m/census/): python -B scripts/launch_06m.py k0
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
from prov import A3V2_DIGEST, CHECKPOINT06L_B_SHA256, LQ_SHA256_06M, RUN06L_B, sha256  # noqa: E402  prompt 06m

PYTHON = str(RC.PYTHON)
TREE = str(RC.G_TREE)
SCRIPTS = str(HERE)
SCRIPT = HERE / "run_06j.py"
K0_UNITS = RC.G_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"  # prompt 06m


def py(script, *args):
    return [PYTHON, "-B", str(HERE / script), *[str(a) for a in args]]


def plan_k0():  # prompt 06m
    tp = f"{TREE}:{SCRIPTS}"
    return {"label": "k0", "stages": [
        {"name": "k0_run", "max_parallel": 1, "jobs": [
            {"name": "41_k0_run_tree", "pythonpath": TREE,
             "argv": py("run_k0.py", "tree", K0_UNITS, "state/k0/tree", "--workers", 6)}]},
        {"name": "k0_compare", "max_parallel": 1, "jobs": [
            {"name": "51_k0_compare_routeB", "pythonpath": tp, "requires": "state/k0/tree/units.jsonl",
             "argv": py("compare_06m.py", "k0", "state/k0/tree", "out/k0/compare_m0.json")}]},
    ]}


PURPOSES = {  # prompt 06m
    "k0": ("levin-phase-floor prompt 06m section 5 on the Mac Studio (mps026298; Python 3.13.16): Gate M0. At the worktree "
           f"{TREE} (Part B's dispatch SHA 38f6a828; code 06m's code commit 48cf87c0's, AdaptiveLevin/levin_quadrature.py "
           f"sha256 {LQ_SHA256_06M[:8]}), with no variant installed: 06k's 100 K0 census units "
           f"(docs/levin-phase-floor/06l/out/k0/units.json, sha256 {sha256(K0_UNITS)[:8]}) re-run with 06l's census harness, "
           f"reading 06's certified scratch copy of A3 v2 read-only (fingerprint {A3V2_DIGEST[:8]}) on a stand-in "
           f"ShardedPool, six workers, 300 s per unit; then each unit compared bit for bit with route B's census "
           f"({RUN06L_B.name}, checkpoint {CHECKPOINT06L_B_SHA256[:8]}). No census. At most six computing processes; "
           "Ray-free, no lock, no store built, nothing written in the tree"),
}
SCOPES = {  # prompt 06m
    "k0": ("docs/levin-phase-floor/06l/out/k0/units.json (100 units, in the tree, read only); outputs under "
           "var/levin-phase-floor-06m/census/: state/k0/tree, out/k0/compare_m0.json, out/cmd/k0"),
}

HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if label not in PURPOSES:
        raise SystemExit(f"usage: launch_06m.py {'|'.join(PURPOSES)}")
    plan = {"k0": plan_k0}[label]()  # prompt 06m
    plan_p = ROOT / "state" / f"plan_{label}.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists; not overwritten")
    plan_p.write_text(json.dumps(plan, indent=1))
    n = sum(len(st["jobs"]) for st in plan["stages"])
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="06m", slug=label, purpose=PURPOSES[label], argv=argv, cwd=str(ROOT),
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
