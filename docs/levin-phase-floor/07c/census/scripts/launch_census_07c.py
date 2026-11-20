"""levin-phase-floor prompt 07c section 6: the census at 07c's code. Copied from 07b's census launch_census_07b.py (a
record, not edited), changed only where marked "prompt 07c": the module checked is h's (d141a80b..., code 67b5af74's),
the prompt is 07c, the purpose names 07c's tree, code commit, harness (07b's, with 07c's three counts and stop
instrument), its over-budget pass, its backstop (48 h) and its before (07b's census); the projection's highest margin is
given on the command line; the launcher refuses if a 07c census run is ``running``, and its scripts' SHA-256 are written
first to state/scripts_sha256_at_census_launch.txt. The launch is 07b's, unchanged. Usage (from
var/levin-phase-floor-07c/census/): python -B scripts/launch_census_07c.py PROJECTION_S HIGHEST_S
07b's text follows.

levin-phase-floor prompt 07b section 6: the census at 07b's code. Copied from 07a's census launch_census_07a.py (a
record, not edited), changed only where marked "prompt 07b": the module checked is h's (8956c7b6..., code 5e19c4b6's),
the prompt is 07b, the purpose names 07b's tree, its harness (07a's, with 07b's two counts and rate-rule instrument), its
over-budget pass, its backstop (48 h) and its before (07a's census), the launcher refuses if a 07b census run is
``running``, and its scripts' SHA-256 are written first to state/scripts_sha256_at_census_launch.txt. The launch is 07a's,
unchanged. Usage (from var/levin-phase-floor-07b/census/): python -B scripts/launch_census_07b.py PROJECTION_S
07a's text follows.

levin-phase-floor prompt 07a section 6: the census h. Copied from 07's census launch_census_07.py (a record, not
edited), changed only where marked "prompt 07a": the module checked is h's (4ea1948b..., code 1ec734d7's), the prompt is
07a, the purpose names 07a's tree, its harness (07's, with section 6's diagnostics), its over-budget pass and its before
(07's census), the launcher refuses if a 07a census run is ``running``, and its scripts' SHA-256 are written first to
state/scripts_sha256_at_census_launch.txt. The launch is 07's, unchanged. Usage (from var/levin-phase-floor-07a/census/):
python -B scripts/launch_census_07a.py PROJECTION_S
07's text follows.

levin-phase-floor prompt 07 section 6: the census g. Copied from 06m's census launch_census_06m.py (a record, not
edited), changed only where marked "prompt 07": the module checked is g's (4e5da371...), the prompt is 07, the purpose
names 07's tree, its harness, its over-budget pass and its before (06m's census), and the launcher refuses if a 07
census run is ``running``. The launch is 06m's, unchanged. Usage (from var/levin-phase-floor-07/census/):
python -B scripts/launch_census_07.py PROJECTION_S
06m's text follows.

levin-phase-floor prompt 06m section 5: the census m. Copied from 06l's launch_census_06l.py (a record, not
edited), changed only where marked "prompt 06m": no variant (run_census.py is given no --variant, so every worker runs
the tree's own module, checked here to be 06m's beaf0e23...); the prompt is 06m and the slug ``census``; the purpose
names 06m's tree; the launcher refuses if a 06m census run is ``running``.
Usage (from var/levin-phase-floor-06m/census/): python -B scripts/launch_census_06m.py PROJECTION_S
06l's text follows.

levin-phase-floor prompt 06l section 3 (Part C): copied from 06k's launch_census_06k.py (a record; its copy is in
state/scripts_06k_reference/), changed only where marked "prompt 06l": the one route is routeB, the prompt is 06l, the
purpose names 06l's tree and variant, and the launcher refuses if any 06l census run is ``running``.
Usage (from var/levin-phase-floor-06l/): python -B scripts/launch_census_06l.py routeB PROJECTION_S
06k's text follows.

levin-phase-floor prompt 06k section 3 (Part C): launch one route's census as a registered, detached run
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
from prov import A3V2_DIGEST, G_SHA, G_TREE, LQ_SHA256_06M, PYTHON, WIDER, sha256  # noqa: E402  prompt 06m: LQ_SHA256_06M
from prov import CHECKPOINT06M_SHA256, LQ_SHA256_07G, RUN06M  # noqa: E402  prompt 07
from prov import CHECKPOINT07_SHA256, H_SHA, H_TREE, LQ_SHA256_07AH, RUN07  # noqa: E402  prompt 07a
from prov import CHECKPOINT07A_SHA256, LQ_SHA256_07BH, RUN07A  # noqa: E402  prompt 07b
from prov import CHECKPOINT07B_SHA256, CODE_SHA_07C, LQ_SHA256_07CH, RUN07B  # noqa: E402  prompt 07c

import runregistrykit  # noqa: E402

SCRIPT = HERE / "run_census.py"
ROWS = WIDER / "state" / "rows.json"
N_ROWS = len(json.loads(ROWS.read_text())["rows"])
ROUTES_06K = {"route1": "route 1 (the noise clause compares delta1 with nu1 = phi_P + phi_L + phi_R)",
          "route2": "route 2 (where rule (b) finds a leaf untrusted, rule (b) on the values corrected by -E)",
          "route2lit": "route 2 literal (every checked leaf decided and declared by rule (b) on the values corrected by -E)",
          "route3": "route 3 (at each stop test the untrusted refinable leaves whose D sum to at most 1e-3 of the request are left)"}


ROUTES = {"routeB": "route B"}  # prompt 06l (kept for the record; not used at 06m)


def main():  # prompt 06m: no variant; the slug is census; 06m's purpose
    if len(sys.argv) != 3:
        raise SystemExit("usage: launch_census_07c.py PROJECTION_S HIGHEST_S")  # prompt 07c
    proj = float(sys.argv[1])
    highest = float(sys.argv[2])  # prompt 07c
    for r in runregistrykit.list_runs():
        rid = getattr(r, "id", None) or (r.get("id") if isinstance(r, dict) else str(r))
        state = getattr(r, "state", None) or (r.get("state") if isinstance(r, dict) else None)
        if "07c-census" in str(rid) and state == "running":  # prompt 07c
            raise SystemExit(f"a 07c census is running: {rid}")
    lq = H_TREE / "AdaptiveLevin" / "levin_quadrature.py"  # prompt 07a
    if sha256(lq) != LQ_SHA256_07CH:  # prompt 07c
        raise SystemExit(f"{lq} is not 07c's module (67b5af74's)")
    sh_p = WIDER / "state" / "scripts_sha256_at_census_launch.txt"  # prompt 07a: the scripts as launched
    if sh_p.exists():
        raise SystemExit(f"{sh_p} exists; not overwritten")
    sh_p.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.sh"))))
    expected = 2 * N_ROWS
    purpose = (f"levin-phase-floor prompt 07c section 6 on the Mac Studio (mps026298; Python 3.13.16): the census at "  # prompt 07c
               f"07c's code (the stop below a refused leaf, variant (S), code commit {CODE_SHA_07C[:8]}), 07b's census re-run "
               f"with no variant installed (the tree's own AdaptiveLevin/levin_quadrature.py, sha256 {LQ_SHA256_07CH[:8]}), "
               f"every A3 v2 QuadSourceIntegral row ({N_ROWS}) at rtol 1e-7 and 1e-8, Ray-free, no lock, with 07b's census "
               f"harness as changed for 07c (each call's num_chase_stops, num_chase_stopped_leaves and "
               f"num_chase_noise_bisections; the stop's instrument: every stopped leaf's route, depth, trust, r, declaration "
               f"and share of its call's abserr; 07a's diagnostics and 07b's rate-rule instrument kept). Tree h: the worktree "
               f"{H_TREE} at Part C's dispatch SHA {H_SHA[:8]}, clean. It reads, read-only, 06's certified scratch copy of "
               f"A3 v2 (fingerprint {A3V2_DIGEST[:8]}) on a stand-in ShardedPool. One worker per k per rtol, at most six at "
               f"once; 300 s per unit, then 06a's over-budget pass (each unit over budget re-run once at 3,600 s, after both "
               f"sets, recorded as <uid>/overbudget-3600). Projected {proj:.0f} s (highest margin {highest:.0f} s); past the "
               f"projection the progress and projected finish are reported every 10 min, and the run is stopped only if the "
               f"projection counting the units in flight is 48 h or more. The before is 07b's census ({RUN07B.name}, "
               f"checkpoint {CHECKPOINT07B_SHA256[:8]}, 45,016 s). Its checkpoint and region records are prompt 07d's before")
    scope = (f"all {N_ROWS} QuadSourceIntegral rows (state/rows.json, sha256 aa523d28) at 1e-7 and 1e-8 at the tree h, no "
             "variant; no store write, no Ray, no lock, nothing written in the tree; then 06a's over-budget pass")  # prompt 07c
    beat = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each worker's "
            "start and every ten units' records, and every 60 s by a thread of the driver while its process lives.")
    run = runregistrykit.begin(campaign="levin-phase-floor", prompt="07c", slug="census", purpose=purpose,  # prompt 07c
                               argv=[str(PYTHON), "-B", str(SCRIPT), "--run-dir", "<run dir>",
                                     "--projection-s", str(proj)],
                               cwd=str(WIDER), script=str(SCRIPT), unit="units", expected_units=expected, checkpoint=True,
                               results=None, scope=scope, heartbeat_means=beat)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    stdout = open(run.stdout_path, "ab")
    stderr = open(run.stderr_path, "ab")
    command = " ".join(shlex.quote(a) for a in [str(PYTHON), "-B", str(SCRIPT), "--run-dir", run.path,
                                                  "--projection-s", str(proj)]) + "; exit $?"
    child = subprocess.Popen(["nohup", "sh", "-c", command], cwd=str(WIDER), env=env, stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    stdout.close()
    stderr.close()
    run.heartbeat(pid=child.pid, stage="launched (detached sh started)")
    print(f"run id      {run.id}\nrun dir     {run.path}\nexpected    {expected} units\nscript      {SCRIPT} sha256 {sha256(SCRIPT)}")
    print(f"module      {lq} sha256 {sha256(lq)} (no variant)\nsh pid      {child.pid}\ncommand     {command}\nlauncher    pid {os.getpid()}, exiting without waiting")


if __name__ == "__main__":
    main()
    sys.exit(0)
