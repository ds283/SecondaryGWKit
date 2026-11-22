"""levin-phase-floor prompt 07a section 6: write Gate W0's plan and launch 06j's driver run_06j.py (07's copy, unchanged) on
it as one registered, detached run (campaign levin-phase-floor, prompt 07a, slug k0). Copied from 07's census
launch_07.py (a record, not edited), with its plans replaced, each change marked "prompt 07a". The launch itself is 07's
(06m's, 06l's, 06k's), unchanged: ``runregistrykit.begin`` first (manifest, script run_06j.py), then
``nohup sh -c '<driver>; exit $?'`` in a new session, stdin from /dev/null, stdout and stderr into the run directory, the
sh pid recorded with ``Run.heartbeat(pid=...)``; the driver replaces it with its own.

Plan ``k0`` (Gate W0, slug ``k0``), stage by stage, each stage after the one before, never more than six computing
processes:
1. ``k0_run_parent``: 06k's 100 K0 units (docs/levin-phase-floor/06l/out/k0/units.json, in the tree h) run by run_k0.py in
   the tree h with the parent's module installed (variant ``parent``: state/variants/levin_quadrature_parent.py, git show
   dadd63a0:..., 4e5da371..., byte-identical to 07's census tree e182bc7a's), the diagnostics on, six workers, into
   state/k0/parent (stop_on_fail: if the run itself fails, nothing further runs);
2. ``k0_compare_parent``: compare_07a.py ``whole`` against 07's census into out/k0/compare_w0b.json (W0's second clause);
3. ``k0_run_h``: the same 100 units at h (h's own module, 4ea1948b..., code 1ec734d7's), six workers, into state/k0/h:
   the census's own unit timings at h for its cost projection, and the units the third clause chooses from;
4. ``k0_compare_h``: compare_07a.py ``calls`` against 07's census into out/k0/compare_k0_h_calls.json (descriptive: the
   classing of section 6's comparison on those units, before the census);
5. ``w0_select``: select_w0_units_07a.py on state/k0/h: the units of the third and fourth clauses, by the rule in its
   docstring (state/w0c3_units.json, state/w0c4_units.json);
6. ``w0c3_run``: W0's third clause, the two units (one with a window, one with a route change) run at h with the
   diagnostics on and, at the same time in fresh workers, off, into state/w0c3/on and state/w0c3/off (at most four
   workers);
7. ``w0c3_compare``: compare_07a.py ``instr`` on the two into out/w0c3/compare_instr.json;
8. ``w0c4_run``: W0's fourth clause, the window unit run at h twice, in two fresh workers at once, into state/w0c4/run1
   and state/w0c4/run2;
9. ``w0c4_compare``: compare_07a.py ``det`` on the two into out/w0c4/compare_det.json.

Plan ``references`` (the user's references at the W5 stop, state/ruling_w5.md item 2; slug ``references``), as 07's plan
``references`` (launch_07.py, the user's ruling at 07's stop 4): one stage, one reading at a time (each reading's tool runs
six worker processes), FCC r1 and r2 on each of the four calls of out/refs/inputs.json, QAWO r1 and r2 on those whose QAWO
readings project at 30 minutes or less (out/refs/refs_projection.json, its ``readings``), each call's readings in the
tool's own order, the calls cheapest first, each reading by 06g's one_reading.py (unchanged) in its own process,
PYTHONPATH the tree h then this directory, into out/refs/readings/<sid>-<impl>-r<res>.json; run_06j.py's run backstop
as 07's (report past 30 minutes, stop at a projected finish of 48 h).

Usage (from var/levin-phase-floor-07a/census/): python -B scripts/launch_07a.py k0|references
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
from prov import (A3V2_DIGEST, CHECKPOINT07_SHA256, H_SHA, H_TREE, LQ_SHA256_07AH, LQ_SHA256_07APARENT,  # noqa: E402
                  PARENT_VARIANT_07A, RUN07, sha256)

PYTHON = str(RC.PYTHON)
TREE = str(H_TREE)  # prompt 07a
SCRIPTS = str(HERE)
SCRIPT = HERE / "run_06j.py"
K0_UNITS = H_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"  # prompt 07a: in the tree h


def py(script, *args):
    return [PYTHON, "-B", str(HERE / script), *[str(a) for a in args]]


def plan_k0():  # prompt 07a
    return {"label": "k0", "stages": [
        {"name": "k0_run_parent", "max_parallel": 1, "stop_on_fail": True, "jobs": [
            {"name": "41_k0_run_parent", "pythonpath": TREE,
             "argv": py("run_k0.py", "parent", K0_UNITS, "state/k0/parent", "--workers", 6, "--tree", "h")}]},
        {"name": "k0_compare_parent", "max_parallel": 1, "jobs": [
            {"name": "51_k0_compare_parent_whole", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": "state/k0/parent/units.jsonl",
             "argv": py("compare_07a.py", "whole", "state/k0/parent", "out/k0/compare_w0b.json", "--processes", 6)}]},
        {"name": "k0_run_h", "max_parallel": 1, "jobs": [
            {"name": "61_k0_run_h", "pythonpath": TREE,
             "argv": py("run_k0.py", "tree", K0_UNITS, "state/k0/h", "--workers", 6, "--tree", "h")}]},
        {"name": "k0_compare_h", "max_parallel": 1, "jobs": [
            {"name": "71_k0_compare_h_calls", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": "state/k0/h/units.jsonl",
             "argv": py("compare_07a.py", "calls", "state/k0/h", "out/k0/compare_k0_h_calls.json", "--processes", 6)}]},
        {"name": "w0_select", "max_parallel": 1, "jobs": [
            {"name": "75_w0_select", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": "state/k0/h/units.jsonl",
             "argv": py("select_w0_units_07a.py", "state/k0/h")}]},
        {"name": "w0c3_run", "max_parallel": 2, "jobs": [
            {"name": "81_w0c3_on", "pythonpath": TREE, "requires": "state/w0c3_units.json",
             "argv": py("run_k0.py", "tree", "state/w0c3_units.json", "state/w0c3/on", "--workers", 2, "--tree", "h")},
            {"name": "82_w0c3_off", "pythonpath": TREE, "requires": "state/w0c3_units.json",
             "argv": py("run_k0.py", "tree", "state/w0c3_units.json", "state/w0c3/off", "--workers", 2, "--tree", "h",
                        "--no-instrument")}]},
        {"name": "w0c3_compare", "max_parallel": 1, "jobs": [
            {"name": "85_w0c3_compare_instr", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": "state/w0c3/off/units.jsonl",
             "argv": py("compare_07a.py", "instr", "state/w0c3/on", "state/w0c3/off", "out/w0c3/compare_instr.json")}]},
        {"name": "w0c4_run", "max_parallel": 2, "jobs": [
            {"name": "91_w0c4_run1", "pythonpath": TREE, "requires": "state/w0c4_units.json",
             "argv": py("run_k0.py", "tree", "state/w0c4_units.json", "state/w0c4/run1", "--workers", 1, "--tree", "h")},
            {"name": "92_w0c4_run2", "pythonpath": TREE, "requires": "state/w0c4_units.json",
             "argv": py("run_k0.py", "tree", "state/w0c4_units.json", "state/w0c4/run2", "--workers", 1, "--tree", "h")}]},
        {"name": "w0c4_compare", "max_parallel": 1, "jobs": [
            {"name": "95_w0c4_compare_det", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": "state/w0c4/run2/units.jsonl",
             "argv": py("compare_07a.py", "det", "state/w0c4/run1", "state/w0c4/run2", "out/w0c4/compare_det.json")}]},
    ]}


BACKSTOP_S = 172800.0  # prompt 07a: as 07's references run


def plan_references():  # prompt 07a: the user's references (state/ruling_w5.md item 2)
    inputs = json.loads((ROOT / "out" / "refs" / "inputs.json").read_text())["inputs"]
    proj = json.loads((ROOT / "out" / "refs" / "refs_projection.json").read_text())["calls"]
    if set(inputs) != set(proj) or len(inputs) != 4:
        raise SystemExit("the inputs and the projection do not name the same four calls")
    cost = {s: sum(proj[s]["projection"][f"{i}_r{r}"]["with_margin_s"] for i, r in proj[s]["readings"]) for s in proj}
    (ROOT / "out" / "refs" / "readings").mkdir(parents=True, exist_ok=True)
    jobs = []
    for sid in sorted(proj, key=lambda s: (cost[s], s)):
        e = inputs[sid]
        for impl, res in proj[sid]["readings"]:
            jobs.append({"name": f"{sid}-{impl}-r{res}", "pythonpath": f"{TREE}:{SCRIPTS}", "requires": e["path"],
                         "projected_s": proj[sid]["projection"][f"{impl}_r{res}"]["with_margin_s"],
                         "argv": py("one_reading.py", e["path"], impl, res, f"out/refs/readings/{sid}-{impl}-r{res}.json",
                                    "--sid", sid, "--unit", e["unit"], "--index", e["index"], "--rtol", e["rtol"],
                                    "--workers", 6)})
    return {"label": "references", "backstop_s": BACKSTOP_S, "check_after_s": 1800.0,
            "stages": [{"name": "readings", "max_parallel": 1, "jobs": jobs}]}


PURPOSES = {  # prompt 07a
    "k0": ("levin-phase-floor prompt 07a section 6 on the Mac Studio (mps026298; Python 3.13.16): Gate W0's second, third "
           f"and fourth clauses. (2) In the worktree {TREE} (h, at Part C's dispatch SHA {H_SHA[:8]}, code 1ec734d7's) with "
           f"the parent's AdaptiveLevin/levin_quadrature.py installed in place of h's by variant_loader (git show dadd63a0, "
           f"sha256 {LQ_SHA256_07APARENT[:8]}, byte-identical to 07's census tree e182bc7a's; {PARENT_VARIANT_07A}): 06k's "
           f"100 K0 census units (docs/levin-phase-floor/06l/out/k0/units.json, sha256 {sha256(K0_UNITS)[:8]}) re-run with "
           "07's census harness as changed for 07a, then each unit compared bit for bit with 07's census "
           f"({RUN07.name}, checkpoint {CHECKPOINT07_SHA256[:8]}) with 07a's new keys set aside. Then the same 100 units "
           f"at h (levin_quadrature.py {LQ_SHA256_07AH[:8]}) for the census's cost projection, compared call by call with "
           "07's; (3) one unit with a window (z12's row, h-1e-8-1-2174) and one with a route change (chosen from the K0 "
           "units at h by select_w0_units_07a.py's rule) run at h with section 6's diagnostics on and off, compared bit "
           "for bit; and (4) the window unit run twice at h in two fresh workers and compared bit for bit. Every unit "
           f"reads 06's certified scratch copy of A3 v2 read-only (fingerprint {A3V2_DIGEST[:8]}) on a stand-in "
           "ShardedPool, 300 s per unit. No census. At most six computing processes; Ray-free, no lock, no store built, "
           "nothing written in a tree"),
}
PURPOSES["references"] = (  # prompt 07a
    "levin-phase-floor prompt 07a section 6 on the Mac Studio (mps026298; Python 3.13.16): the user's references at the W5 "
    "stop (state/ruling_w5.md: 'Yes, W5 gated only, and take references for both calls'), by 07's V4 method: "
    "tools/filon_reference.py's readings (d37e633b, as 07's) of rows 3-678 and 1-34599, group calls 2 (G+q-r) and 3 (G-q+r), "
    "captured at h and checked bit for bit against 07a's census and, with the parent's module, against 07's "
    "(out/refs/check_captures_h.json, out/refs/check_captures_07.json; one integrand per row and index, the same at both "
    "rtols, out/refs/integrands.json); FCC r1 and r2 on all four, QAWO r1 and r2 where projected at 30 minutes or less "
    "(out/refs/refs_projection.json, 737 s in all with margin); one reading at a time, six worker processes each, one "
    "process per reading (06g's one_reading.py); stopped only at a projected finish of 48 h. Ray-free, no store opened, "
    "nothing written in a tree")
SCOPES = {  # prompt 07a
    "k0": ("docs/levin-phase-floor/06l/out/k0/units.json (100 units, in the tree h, read only), state/w0c3_units.json (2 "
           "units) and state/w0c4_units.json (1 unit), the last two written by the run's stage 5; outputs under "
           "var/levin-phase-floor-07a/census/: state/k0/parent, state/k0/h, state/w0c3/on, state/w0c3/off, state/w0c4/run1, "
           "state/w0c4/run2, out/k0/, out/w0c3/, out/w0c4/, out/cmd/k0"),
}

SCOPES["references"] = ("4 captured calls (out/refs/inputs.json), up to 16 readings; outputs out/refs/readings/, "  # prompt 07a
                         "out/cmd/references")
HEARTBEAT_MEANS = ("Refreshed by the launcher once with the sh pid, by the driver at start with its own pid, at each "
                   "command's start and end, and every 60 s by a thread of the driver while its process lives.")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else None
    if label not in PURPOSES:
        raise SystemExit(f"usage: launch_07a.py {'|'.join(PURPOSES)}")
    plan = {"k0": plan_k0, "references": plan_references}[label]()  # prompt 07a
    plan_p = ROOT / "state" / f"plan_{label}.json"
    if plan_p.exists():
        raise SystemExit(f"{plan_p} exists; not overwritten")
    plan_p.write_text(json.dumps(plan, indent=1))
    n = sum(len(st["jobs"]) for st in plan["stages"])
    argv = [PYTHON, "-B", str(SCRIPT), "--run-dir", "<run dir>", "--plan", str(plan_p)]
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="07a", slug=label, purpose=PURPOSES[label], argv=argv, cwd=str(ROOT),
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
