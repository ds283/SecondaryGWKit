"""levin-phase-floor prompt 06k section 3 (Part C): launch one route's census as a registered, detached run
(campaign levin-phase-floor, prompt 06k, slug census-<route>). New in 06k, after 06g's census scripts/launch.py:
``runregistrykit.begin`` first (manifest, script run_census.py), then ``nohup sh -c '<driver>; exit $?'`` in a
new session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid recorded with
``Run.heartbeat(pid=...)``; the driver (run_census.py --run-dir RUN --variant ROUTE --projection-s P) replaces
it with its own. One census at a time: the launcher refuses if a 06k census run is ``running``.

Usage (from var/levin-phase-floor-06k/): python -B scripts/launch_census_06k.py ROUTE PROJECTION_S
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import A3V2_DIGEST, G_SHA, G_TREE, PYTHON, WIDER, sha256  # noqa: E402

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_census.py"
ROWS = WIDER / "state" / "rows.json"
N_ROWS = len(json.loads(ROWS.read_text())["rows"])
ROUTES = {"route1": "route 1 (the noise clause compares delta1 with nu1 = phi_P + phi_L + phi_R)",
          "route2": "route 2 (where rule (b) finds a leaf untrusted, rule (b) on the values corrected by -E)",
          "route2lit": "route 2 literal (every checked leaf decided and declared by rule (b) on the values corrected by -E)",
          "route3": "route 3 (at each stop test the untrusted refinable leaves whose D sum to at most 1e-3 of the request are left)"}


def main():
    route, proj = sys.argv[1], float(sys.argv[2])
    if route not in ROUTES:
        raise SystemExit(f"usage: launch_census_06k.py {'|'.join(ROUTES)} PROJECTION_S")
    for r in runregistrykit.list_runs():
        rid = getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else str(r))
        state = getattr(r, "state", None) or (r.get("state") if isinstance(r, dict) else None)
        if "06k-census" in str(rid) and state == "running":
            raise SystemExit(f"a 06k census is running: {rid}")
    vp = HERE / "variants" / f"levin_quadrature_{route}.py"
    expected = 2 * N_ROWS
    purpose = (f"levin-phase-floor prompt 06k Part C on the Mac Studio (mps026298; Python 3.13.16): 06g's census re-run "
               f"at the tree with {ROUTES[route]} loaded in place of AdaptiveLevin/levin_quadrature.py "
               f"(scripts/variants/levin_quadrature_{route}.py, sha256 {sha256(vp)[:12]}), every A3 v2 QuadSourceIntegral "
               f"row ({N_ROWS}) at rtol 1e-7 and 1e-8, Ray-free, no lock, with 06g's census harness and records plus the "
               f"route's firings per call. Tree g: the worktree {G_TREE} at 06k's dispatch SHA {G_SHA[:8]} (production "
               f"code 1c5336e's). It reads, read-only, 06's certified scratch copy of A3 v2 (fingerprint {A3V2_DIGEST[:8]}) "
               f"on a stand-in ShardedPool. One worker per k per rtol, at most six at once; 300 s per unit. Projected "
               f"{proj:.0f} s; past it the progress and projected finish are reported every 10 min, and the run is stopped "
               f"only if the projection counting the units in flight is a day or more. 06g's census took 5 h 40 min, 06h's 2 h 33 min")
    scope = (f"all {N_ROWS} QuadSourceIntegral rows (state/rows.json, sha256 aa523d28) at 1e-7 and 1e-8 under {route}; "
             "no store write, no Ray, no lock, no code changed in the tree")
    beat = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each worker's "
            "start and every ten units' records, and every 60 s by a thread of the driver while its process lives.")
    run = runregistrykit.begin(campaign="levin-phase-floor", prompt="06k", slug=f"census-{route}", purpose=purpose,
                               argv=[str(PYTHON), "-B", str(SCRIPT), "--run-dir", "<run dir>", "--variant", route,
                                     "--projection-s", str(proj)],
                               cwd=str(WIDER), script=str(SCRIPT), unit="units", expected_units=expected, checkpoint=True,
                               results=None, scope=scope, heartbeat_means=beat)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    command = " ".join(shlex.quote(a) for a in [str(PYTHON), "-B", str(SCRIPT), "--run-dir", run.path, "--variant", route,
                                                  "--projection-s", str(proj)]) + "; exit $?"
    child = subprocess.Popen(["nohup", "sh", "-c", command], cwd=str(WIDER), env=env, stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    stdout.close()
    stderr.close()
    run.heartbeat(pid=child.pid, stage="launched (detached sh started)")
    print(f"run id      {run.id}\nrun dir     {run.path}\nexpected    {expected} units\nscript      {SCRIPT} sha256 {sha256(SCRIPT)}")
    print(f"variant     {vp} sha256 {sha256(vp)}\nsh pid      {child.pid}\ncommand     {command}\nlauncher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
