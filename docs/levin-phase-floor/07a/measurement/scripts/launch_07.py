"""levin-phase-floor prompt 07 section 6: write a run's plan and launch 06j's driver run_06j.py (copied unchanged) on it as
one registered, detached run (campaign levin-phase-floor, prompt 07). Copied from 06m's census launch_06m.py (a record,
not edited), with its plan replaced, each change marked "prompt 07". The launch itself is 06m's (06l's, 06k's),
unchanged: ``runregistrykit.begin`` first (manifest, script run_06j.py), then ``nohup sh -c '<driver>; exit $?'`` in a
new session, stdin from /dev/null, stdout and stderr into the run directory, the sh pid recorded with
``Run.heartbeat(pid=...)``; the driver replaces it with its own.

Plan ``k0`` (Gate V0, slug ``k0``), stage by stage, each stage after the one before, never more than six computing
processes:
1. ``k0_run_p``: 06k's 100 K0 units (docs/levin-phase-floor/06l/out/k0/units.json, in the tree g) run by run_k0.py at
   the parent p (the worktree at 91472dbe, 06m's code, AdaptiveLevin/levin_quadrature.py beaf0e23...), no variant, six
   workers, into state/k0/p (stop_on_fail: if the run itself fails, nothing further runs);
2. ``k0_compare_p``: compare_07.py ``whole`` against 06m's census into out/k0/compare_v0b.json (V0's second clause);
3. ``k0_run_g``: the same 100 units at g (the worktree at e182bc7a, code adffa21b's, 4e5da371...), six workers, into
   state/k0/g: the census's own unit timings at g for its cost projection, paired with 06m's (not a clause of V0);
4. ``k0_compare_g``: compare_07.py ``calls`` against 06m's census into out/k0/compare_k0_g_calls.json (descriptive:
   V1's comparison on those units, before the census);
5. ``v0c_run``: V0's third clause, the unit of state/v0c_units.json (z12's row at 1e-8) run at g twice, in two fresh
   workers at once (two processes), into state/v0c/run1 and state/v0c/run2;
6. ``v0c_compare``: compare_07.py ``det`` on the two into out/v0c/compare_det.json.

Plan ``references`` (V4's references, slug ``references``), as the user ruled at stop 4 (state/
v4_reading_list_20261109.json, its SHA-256 checked): one stage, one reading at a time (each reading's tool runs six
worker processes), 74 commands: FCC r1 and r2 on all 24 calls, QAWO r1 and r2 on the 13 listed, each call's readings in
the tool's own order (QAWO r1, r2, FCC r1, r2), the calls cheapest first by the projection
(out/v4/refs_projection.json), each reading by 06g's one_reading.py (06j's copy, unchanged) in its own process,
PYTHONPATH the tree g then this directory, into out/v4/readings/<sid>-<impl>-r<res>.json. A reading past 30 minutes is
not stopped (06g's ruling: a check, not a cutoff); run_06j.py's run backstop (prompt 07) reports and stops the run only
at a projected finish of 48 h. The calibration capture (z12's) is not read.

Usage (from var/levin-phase-floor-07/census/): python -B scripts/launch_07.py k0|references
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
from prov import A3V2_DIGEST, CHECKPOINT06M_SHA256, LQ_SHA256_07G, LQ_SHA256_07P, RUN06M, sha256  # noqa: E402  prompt 07

PYTHON = str(RC.PYTHON)
TREE = str(RC.G_TREE)
PTREE = str(RC.P_TREE)  # prompt 07
SCRIPTS = str(HERE)
SCRIPT = HERE / "run_06j.py"
K0_UNITS = RC.G_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"  # prompt 06m
V0C_UNITS = ROOT / "state" / "v0c_units.json"  # prompt 07


def py(script, *args):
    return [PYTHON, "-B", str(HERE / script), *[str(a) for a in args]]


def plan_k0():  # prompt 07
    return {"label": "k0", "stages": [
        {"name": "k0_run_p", "max_parallel": 1, "stop_on_fail": True, "jobs": [
            {"name": "41_k0_run_p", "pythonpath": PTREE,
             "argv": py("run_k0.py", "tree", K0_UNITS, "state/k0/p", "--workers", 6, "--tree", "p")}]},
        {"name": "k0_compare_p", "max_parallel": 1, "jobs": [
            {"name": "51_k0_compare_p_whole", "pythonpath": f"{PTREE}:{SCRIPTS}", "requires": "state/k0/p/units.jsonl",
             "argv": py("compare_07.py", "whole", "state/k0/p", "out/k0/compare_v0b.json", "--processes", 6)}]},
        {"name": "k0_run_g", "max_parallel": 1, "jobs": [
            {"name": "61_k0_run_g", "pythonpath": TREE,
             "argv": py("run_k0.py", "tree", K0_UNITS, "state/k0/g", "--workers", 6, "--tree", "g")}]},
        {"name": "k0_compare_g", "max_parallel": 1, "jobs": [
            {"name": "71_k0_compare_g_calls", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": "state/k0/g/units.jsonl",
             "argv": py("compare_07.py", "calls", "state/k0/g", "out/k0/compare_k0_g_calls.json", "--processes", 6)}]},
        {"name": "v0c_run", "max_parallel": 2, "jobs": [
            {"name": "81_v0c_run1", "pythonpath": TREE,
             "argv": py("run_k0.py", "tree", V0C_UNITS, "state/v0c/run1", "--workers", 1, "--tree", "g")},
            {"name": "82_v0c_run2", "pythonpath": TREE,
             "argv": py("run_k0.py", "tree", V0C_UNITS, "state/v0c/run2", "--workers", 1, "--tree", "g")}]},
        {"name": "v0c_compare", "max_parallel": 1, "jobs": [
            {"name": "91_v0c_compare_det", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": "state/v0c/run2/units.jsonl",
             "argv": py("compare_07.py", "det", "state/v0c/run1", "state/v0c/run2", "out/v0c/compare_det.json")}]},
    ]}


READING_LIST = ROOT / "state" / "v4_reading_list_20261109.json"  # prompt 07: the user's ruling at stop 4
READING_LIST_SHA256 = "ce594095f08e0010a955fcd8c4a7027acd905662350850616ddfe9217a9f6e82"
BACKSTOP_S = 172800.0  # prompt 07: the run is stopped if its projected finish passes 48 h (the ruling at stop 4)


def plan_references():  # prompt 07: V4's references, the subset ruled at stop 4
    if sha256(READING_LIST) != READING_LIST_SHA256:
        raise SystemExit("the reading list has another SHA-256")
    lst = json.loads(READING_LIST.read_text())
    qawo = {c["sid"] for c in lst["qawo_calls"]}
    inputs = json.loads((ROOT / "out" / "v4" / "inputs.json").read_text())["inputs"]
    proj = {c["sid"]: c["readings_s"] for c in json.loads((ROOT / "out" / "v4" / "refs_projection.json").read_text())["calls"]}
    if sha256(ROOT / "out" / "v4" / "refs_projection.json") != lst["projection_sha256"]:
        raise SystemExit("the projection is not the one the reading list was made from")
    kinds = {s: ([("qawo", 1), ("qawo", 2)] if s in qawo else []) + [("fcc", 1), ("fcc", 2)] for s in lst["fcc_calls"]}
    cost = {s: sum(proj[s][f"{i}_r{r}"] for i, r in kinds[s]) for s in kinds}
    sids = sorted(kinds, key=lambda s: (cost[s], s))
    if len(sids) != 24 or len(qawo) != 13:
        raise SystemExit(f"{len(sids)} calls, {len(qawo)} with QAWO")
    (ROOT / "out" / "v4" / "readings").mkdir(parents=True, exist_ok=True)
    jobs = []
    for sid in sids:
        e = inputs[sid]
        for impl, res in kinds[sid]:
            jobs.append({"name": f"{sid}-{impl}-r{res}", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": e["path"],
                         "projected_s": proj[sid][f"{impl}_r{res}"],
                         "argv": py("one_reading.py", e["path"], impl, res, f"out/v4/readings/{sid}-{impl}-r{res}.json",
                                    "--sid", sid, "--unit", e["unit"], "--index", e["index"], "--rtol", e["rtol"],
                                    "--workers", 6)})
    return {"label": "references", "backstop_s": BACKSTOP_S, "check_after_s": 1800.0,
            "reading_list": str(READING_LIST), "reading_list_sha256": READING_LIST_SHA256,
            "stages": [{"name": "readings", "max_parallel": 1, "jobs": jobs}]}


PURPOSES = {  # prompt 07
    "k0": ("levin-phase-floor prompt 07 section 6 on the Mac Studio (mps026298; Python 3.13.16): Gate V0's second and "
           f"third clauses. (b) At the parent worktree {PTREE} (91472dbe, 07's code commit ad732887's parent; 06m's code, "
           f"AdaptiveLevin/levin_quadrature.py sha256 {LQ_SHA256_07P[:8]}), with no variant installed: 06k's 100 K0 census "
           f"units (docs/levin-phase-floor/06l/out/k0/units.json, sha256 {sha256(K0_UNITS)[:8]}) re-run with 06m's census "
           "harness as changed for 07, then each unit compared bit for bit with 06m's census "
           f"({RUN06M.name}, checkpoint {CHECKPOINT06M_SHA256[:8]}) with 07's new keys set aside. Then the same 100 units "
           f"at the tree g {TREE} (e182bc7a, code adffa21b's, levin_quadrature.py {LQ_SHA256_07G[:8]}) for the census's "
           "cost projection, compared call by call with 06m's; and (c) one g unit on a stationary-point row (z12's, "
           "g-1e-8-1-2174) run twice at g in two fresh workers and compared bit for bit. Every unit reads 06's certified "
           f"scratch copy of A3 v2 read-only (fingerprint {A3V2_DIGEST[:8]}) on a stand-in ShardedPool, 300 s per unit. "
           "No census. At most six computing processes; Ray-free, no lock, no store built, nothing written in a tree"),
}
PURPOSES["references"] = (  # prompt 07
    "levin-phase-floor prompt 07 section 6 (V4) on the Mac Studio (mps026298; Python 3.13.16): tools/filon_reference.py's "
    "readings, as ruled at stop 4: FCC r1 and r2 (r2 the reference) on all of V4's 24 stationary-point group calls, QAWO "
    "r1 and r2 on the 13 whose QAWO readings project at 30 minutes or less (state/v4_reading_list_20261109.json ce594095), "
    "74 readings, projected 89,230 s, stopped only at a projected finish of 48 h; the calls "
    f"(state/predictions.md, state/v4_selection.json 15a4391b), captured at the tree g {TREE} (e182bc7a, code "
    "adffa21b's; the tool is b5626ecb's) and each checked bit for bit against 07's census "
    "levin-phase-floor-07-census-20261108T171137; one reading at a time, six worker processes each, one process per reading "
    "(06g's one_reading.py); the 30-minute reading limit is a check, not a cutoff. Ray-free, no store opened, nothing "
    "written in a tree")
SCOPES = {  # prompt 07
    "references": "24 captured calls: FCC r1, r2 on all, QAWO r1, r2 on 13 (74 readings; out/v4/inputs.json); outputs out/v4/readings/, out/cmd/references",
    "k0": ("docs/levin-phase-floor/06l/out/k0/units.json (100 units, in the tree g, read only) and state/v0c_units.json "
           "(1 unit); outputs under var/levin-phase-floor-07/census/: state/k0/p, state/k0/g, state/v0c/run1, "
           "state/v0c/run2, out/k0/, out/v0c/, out/cmd/k0"),
}

HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if label not in PURPOSES:
        raise SystemExit(f"usage: launch_07.py {'|'.join(PURPOSES)}")
    plan = {"k0": plan_k0, "references": plan_references}[label]()  # prompt 07
    plan_p = ROOT / "state" / f"plan_{label}.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists; not overwritten")
    plan_p.write_text(json.dumps(plan, indent=1))
    n = sum(len(st["jobs"]) for st in plan["stages"])
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="07", slug=label, purpose=PURPOSES[label], argv=argv, cwd=str(ROOT),
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
