"""levin-phase-floor prompt 07b section 5 (Part C), the R31 rehearsal agent: write a plan and launch Part A's plan driver
run_06j.py (07b measurement's copy, unchanged, scripts/run_06j.py) on it as one registered, detached run (campaign
levin-phase-floor, prompt 07b). After 07a census's launch_07a.py (a record, not edited): ``runregistrykit.begin`` first
(manifest, script run_06j.py), then ``nohup sh -c '<driver>; exit $?'`` in a new session, stdin from /dev/null, stdout and
stderr into the run directory, the sh pid recorded with ``Run.heartbeat(pid=...)``; the driver replaces it with its own.
New in 07b's rehearsal; only the plans and purposes are this agent's.

Plan ``r31-replay`` (slug ``r31-replay``): the replay of R31's located rows, P1's, the w rows, the block rows and the
rows of the records of interest, at the tree h (0a103b5a, code 5e19c4b6), by the capture harness copied from R30's
(scripts/capture/, changed where marked "prompt 07b r31"), then its check against R31, the four-block check's pass 2 and
check, and the capture of the group records of interest and its check. Each stage one command, ``stop_on_fail``. The
harness runs at most six worker processes at once. Ray-free; under the held rehearsal lock (no Ray is started); no
store written; nothing written in a tree.

Plan ``r31-refs`` (slug ``r31-refs``; added after the user's word at stop 1, "settle which store understates", and
the orchestrator's go-ahead for the seven records only; state/refs_plan_r31.md, written before any reading): for each of
the 7 group records whose |v31 - v30| > e30 + e31, the tool's own four readings (``python -B -m tools.filon_reference
PICKLE --tolerance T --unpickler ray.cloudpickle --workers 6 --out out/refs/readings/<sid>.json``; the tree's tool,
d37e633b...), T = min(e30, e31)/100 in the call's units (from out/refs/refcap_replay_{h,a07}.json), PYTHONPATH the tree
then scripts/capture; one call at a time, the cheapest first (out/refs/refs_projection_r31.json), each job's projection
the sum of its four readings' with margin; the run backstop at 4 h. Ray-free, no lock, no store, nothing in a tree.

Usage (any cwd): python -B launch_r31.py r31-replay|r31-refs
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # var/levin-phase-floor-07b/rehearsal
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
PYTHON = str(MAIN / "venv" / "bin" / "python")
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5"
SCRIPT = HERE / "run_06j.py"
RP = HERE / "capture" / "replay07b_r31.py"
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
    return {"label": "r31-replay", "stages": [
        {"name": n, "max_parallel": 1, "stop_on_fail": True,
         "jobs": [{"name": f"{i:02d}_{n}", "pythonpath": TREE, "argv": argv}]}
        for i, (n, argv) in enumerate(stages, 1)]}


def plan_refs():  # the seven records' references (state/refs_plan_r31.md)
    h = json.loads((ROOT / "out" / "refs" / "refcap_replay_h.json").read_text())["calls"]
    a = json.loads((ROOT / "out" / "refs" / "refcap_replay_a07.json").read_text())["calls"]
    proj = json.loads((ROOT / "out" / "refs" / "refs_projection_r31.json").read_text())["calls"]
    if not (all(x["reproduces"] for x in h) and all(x["reproduces"] for x in a)):
        raise SystemExit("a capture does not reproduce its store")
    (ROOT / "out" / "refs" / "readings").mkdir(parents=True, exist_ok=True)
    jobs = []
    for x, y, p in zip(h, a, proj):
        if not (x["key"] == y["key"] == p["key"] and x["record"] == y["record"] == p["record"]):
            raise SystemExit("the replays and the projection do not list the same calls")
        if not p["in_the_seven"]:
            continue  # the eighth (the converged change) is not read: the user asked about the seven
        e30, e31 = float.fromhex(y["got"]["abserr"]), float.fromhex(x["got"]["abserr"])
        tol = min(e30, e31) / 100.0
        sid = Path(x["pickle"]).stem
        ps = sum(p["projection"][k]["with_margin_s"] for k in ("qawo_r1", "qawo_r2", "fcc_r1", "fcc_r2"))
        jobs.append({"name": sid, "pythonpath": f"{TREE}:{HERE / 'capture'}", "requires": str(Path(x["pickle"]).relative_to(ROOT)),
                     "projected_s": ps, "tolerance_hex": tol.hex(),
                     "argv": [PYTHON, "-B", "-m", "tools.filon_reference", x["pickle"], "--tolerance", repr(tol),
                              "--unpickler", "ray.cloudpickle", "--workers", "6",
                              "--out", str(ROOT / "out" / "refs" / "readings" / f"{sid}.json")]})
    if len(jobs) != 7:
        raise SystemExit(f"{len(jobs)} calls, not 7")
    jobs.sort(key=lambda j: (j["projected_s"], j["name"]))
    return {"label": "r31-refs", "backstop_s": 14400.0, "check_after_s": 1800.0,
            "stages": [{"name": "readings", "max_parallel": 1, "jobs": jobs}]}


PURPOSES = {
    "r31-replay": (
        "levin-phase-floor prompt 07b section 5 (Part C), the R31 rehearsal agent, on the Mac Studio (mps026298; Python "
        "3.13.16): the replay behind S1, S2 and S4 and the refusal counts, as 07a's rehearsal replayed R30. R30's capture "
        "harness (scripts/capture/, changed where marked 'prompt 07b r31': tree h = the worktree "
        f"{TREE} at 0a103b5a, code 5e19c4b6; R31's store read-only through state/stores/r31; each call's "
        "num_edge_rule_refusals and num_range_rule_refusals recorded) re-runs 208 R31 rows (202 located, P1, the w rows, "
        "the two block rows, the 8 records of interest: |v31 - v30| > e30 + e31 or converged changed) and checks every "
        "record bit for bit against R31; then the four S4 blocks' pass 2 and check (each capture against R30's), and the "
        "capture of the 7 group records of interest, checked against R31. Ray-free (never initialised), at most six "
        "worker processes, under the held rehearsal lock, no store written, nothing written in a tree"),
}
PURPOSES["r31-refs"] = (
    "levin-phase-floor prompt 07b section 5 (Part C), the R31 rehearsal agent, on the Mac Studio (mps026298; Python 3.13.16): "
    "the user's word at stop 1, 'settle which store understates' (descriptive, not scored against any rule): references for "
    "the 7 QuadSourceIntegral group records (rtol 1e-7, Route B) whose |v31 - v30| > e30 + e31, by 07's V4 method, the "
    "tool's own four readings (tools/filon_reference.py d37e633b at the tree 0a103b5a: QAWO and FCC at resolutions 1 and 2; "
    "the reference FCC r2, its stated error the readings' scatter, T = min(e30, e31)/100). Each capture (the r31-replay "
    "run's refcap stage) is checked bit for bit against R31 at the tree and against R30 at 07a's module "
    "(out/refs/refcap_replay_{h,a07}.json). One call at a time, six worker processes; projected 842 s with margin "
    "(0.35 h x1.5); backstop 4 h. Ray-free, no lock, no store opened, nothing written in a tree; the decision rule is "
    "state/refs_plan_r31.md, written before any reading")
SCOPES = {
    "r31-replay": ("state/replay_spec.json (208 units), state/blocks2_spec.json, state/refcap_spec.json; outputs under "
                   "var/levin-phase-floor-07b/rehearsal/: state/capture/{replay,blocks2,refcap}, state/inputs/, "
                   "out/replay_calls.csv, out/replay_calls.checks.json, out/blocks.json, out/refcap.json, out/cmd/r31-replay"),
}
SCOPES_REFS = ("7 captured calls (state/inputs/ref-*.pkl, out/refcap.json), 28 readings; outputs out/refs/readings/, "
               "out/cmd/r31-refs, out/r31-refs_results.json")
HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if label not in PURPOSES:
        raise SystemExit(f"usage: launch_r31.py {'|'.join(PURPOSES)}")
    plan = {"r31-replay": plan_replay, "r31-refs": plan_refs}[label]()
    plan_p = ROOT / "state" / f"plan_{label}.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists; not overwritten")
    plan_p.write_text(json.dumps(plan, indent=1))
    n = sum(len(st["jobs"]) for st in plan["stages"])
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="07b", slug=label, purpose=PURPOSES[label], argv=argv, cwd=str(ROOT),
        script=str(SCRIPT), unit="commands", expected_units=n, checkpoint=True, results=None,
        scope=SCOPES_REFS if label == "r31-refs" else SCOPES[label],
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
