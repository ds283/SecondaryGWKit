"""
levin-phase-floor prompt 06 section 6 (copied from prompt 05a's scripts/run_wider.py,
docs/levin-phase-floor/05a/wider/scripts/run_wider.py, a record that is not edited). Changed here,
each change marked "prompt 06", only in its trees, its unit sets and the scripts it hashes (prompt
06 section 6): the trees are ``t5`` (05a's t, the worktree at ``9f66622``) and ``s`` (the code
commit, the worktree at ``f5cee4c``) in place of 05a's ``c`` and ``t``, so the units are

    1. t5 at rtol 1e-7, on every row          (the baseline, and each row's budget)
    2. s at rtol 1e-7, on every row           (the producers re-run: "rerun": true)
    3. s at rtol 1e-8, on the 31 depth-20 rows (the producers re-run: "rerun": true)

with the budget of sets 2 and 3 3x the row's ``t5`` wall time, raised from the dispatcher as
before; and the phase probe (phase_probe06.py) is carried by the rtol 1e-7 units of 04's rows
R1-R7 only, in both trees. 05a's text follows, unchanged; read its ``c`` as ``t5`` and its ``t``
as ``s``.

levin-phase-floor prompt 05a section 6 (copied from prompt 05's scripts/run_wider.py,
docs/levin-phase-floor/05/wider/scripts/run_wider.py, sha256 491d2608..., a record that is not
edited). Changed here, each change marked "prompt 05a", only in its trees and its unit sets
(prompt 05a section 6): the trees are ``c`` (05's c, the worktree at ``c4eaefd``) and ``t`` (the
code commit, the main checkout) in place of 05's ``p`` and ``c``, so the units are

    1. c at rtol 1e-7, on every row          (the baseline, and each row's budget)
    2. t at rtol 1e-7, on every row
    3. t at rtol 1e-8, on the depth-20 rows

with the budget of sets 2 and 3 3x the row's ``c`` wall time; and the rtol 1e-7 units of both
trees carry ``"probe": true`` in their spec, for the phase probe (probe.py; prompt 05a section 6
with section 9), which is also added to the hashed scripts. Prompt 05's text follows, unchanged;
read its ``p`` as this run's ``c`` and its ``c`` as ``t``.

levin-phase-floor prompt 05 section 6: the registered run (slug ``wider``). The driver.

It imports nothing from either tree. It runs prompt 05 section 6's units, in its order, one
checkpoint per (tree, rtol, row):

    1. p at rtol 1e-7, on every row of state/sample.json      (the baseline, and each row's budget)
    2. c at rtol 1e-7, on every row
    3. c at rtol 1e-8, on the depth-20 rows

by starting one worker (worker.py) per tree, in turn, with ``PYTHONPATH`` that tree, ``-B`` and
``PYTHONDONTWRITEBYTECODE=1`` (nothing is written in either tree), and ``OMP_NUM_THREADS=1`` (as 03
and 04). The ``p`` worker runs set 1; then the ``c`` worker runs sets 2 and 3. Each worker writes
one JSON line per unit to a pipe; the driver records each as the run's checkpoint
(``Run.record``), whose record the registry stamps with this script's SHA-256, and whose payload
carries the tree's ``HEAD`` and ``git status --short`` (taken by the worker at the end of the unit)
and the SHA-256 of every script.

**The budget** of a unit of sets 2 and 3 is 3 x its row's ``p`` wall time (set 1), passed to the
``c`` worker, whose dispatcher raises ``OverBudget`` from ``theta_mod_2pi``'s entry once past it
(04's mechanism; no signal). A row whose ``p`` unit did not finish ``done`` has no budget, and that
is recorded.

A heartbeat thread refreshes the registry heartbeat every 60 s. On SIGTERM or SIGINT the driver
records ``killed``, sends SIGTERM to the worker it is running, and dies by the signal.

Usage: python -B run_wider.py --run-dir RUN_DIR       (the registered run; launch.py starts it)
       python -B run_wider.py --out-dir DIR [--rows N1,N2] [--sets 1,2,3]   (a smoke test)
"""

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import PYTHON, TREES, WIDER, sha256  # noqa: E402

BUDGET_FACTOR = 3.0
SCRIPTS = [
    "run_wider.py",
    "worker.py",
    "dispatch.py",
    "harness.py",
    "rebuild.py",
    "prov.py",
    "launch.py",
    "rerun.py",  # prompt 06
    "phase_probe06.py",  # prompt 06
]
PROBE_ROWS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")  # prompt 06: 04's seven rows


def plan(sample, rows=None, sets=(1, 2, 3)):
    rs = sample["rows"]
    if rows is not None:
        rs = [r for r in rs if r["name"] in rows]
    units = []

    def unit(tree, rtol, r):
        return {
            "uid": f"{tree}-{rtol}-{r['name']}",
            "tree": tree,
            "rtol": rtol,
            "row": r["name"],
            "shard": r["shard"],
            "serial": r["serial"],
            "role": r["role"],
            "budget": None,
            "set": None,
        }

    # prompt 06: trees t5, s (05a's c, t); s re-runs the producers; the rtol 1e-7 units of R1-R7
    # carry the phase probe
    if 1 in sets:
        for r in rs:
            u = unit("t5", "1e-7", r)
            u["set"] = 1
            u["probe"] = r["name"] in PROBE_ROWS
            u["rerun"] = False
            units.append(u)
    if 2 in sets:
        for r in rs:
            u = unit("s", "1e-7", r)
            u["set"] = 2
            u["probe"] = r["name"] in PROBE_ROWS
            u["rerun"] = True
            units.append(u)
    if 3 in sets:
        for r in rs:
            if r["role"] == "d20":
                u = unit("s", "1e-8", r)
                u["set"] = 3
                u["probe"] = False
                u["rerun"] = True
                units.append(u)
    return units


def worker_env(tree_name):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TREES[tree_name][0])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-dir")
    g.add_argument("--run-dir")
    parser.add_argument("--rows", default=None, help="smoke test only")
    parser.add_argument("--sets", default="1,2,3", help="smoke test only")
    parser.add_argument("--budget-override", type=float, default=None, help="smoke test only")
    args = parser.parse_args()

    sample = json.loads((WIDER / "state" / "sample.json").read_text())
    rows = None if args.rows is None else args.rows.split(",")
    sets = tuple(int(s) for s in args.sets.split(","))
    units = plan(sample, rows, sets)

    run = None
    current = {"child": None}
    if args.run_dir is not None:
        if args.rows or args.sets != "1,2,3" or args.budget_override is not None:
            raise SystemExit("--rows/--sets/--budget-override are for the smoke test only")
        import runregistrykit

        run = runregistrykit.Run(args.run_dir)
        out_dir = Path(run.path)

        def terminal(signum, _frame):
            child = current["child"]
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGTERM)
                except Exception:
                    pass
            run.finish("killed", exit_code=128 + signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminal)
        run.heartbeat(pid=os.getpid(), stage="started: the t5 worker is next")  # prompt 06

        stop_beat = threading.Event()

        def beat():
            while not stop_beat.wait(60.0):
                try:
                    run.heartbeat()
                except Exception:
                    pass

        threading.Thread(target=beat, daemon=True).start()
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    script_hashes = {name: sha256(HERE / name) for name in SCRIPTS}
    header = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_hashes["run_wider.py"],
        "scripts_sha256": script_hashes,
        "sample_sha256": sha256(WIDER / "state" / "sample.json"),
        "trees": {k: [str(v[0]), v[1]] for k, v in TREES.items()},
        "host": socket.gethostname(),
        "python": sys.version,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "units": [u["uid"] for u in units],
        "budget_factor": BUDGET_FACTOR,
        "budget_override": args.budget_override,
    }
    print("# run_wider header " + json.dumps(header, default=str), flush=True)

    p_wall = {}
    p_status = {}
    seen = []
    problems = []

    def record(uid, data):
        if run is not None:
            run.record(uid, data)
            run.heartbeat(stage=f"unit {uid} {data['status']}")
        else:
            with open(out_dir / "ledger.jsonl", "a") as f:
                f.write(json.dumps({"unit": uid, "data": data}) + "\n")

    def run_worker(tree_name, tree_units):
        if not tree_units:
            return
        spec = {
            "tree": tree_name,
            "units": tree_units,
            "out_dir": str(out_dir),
            "emit_fd": None,
            "emit_file": None,
            "pickle_regions": True,
        }
        r_fd, w_fd = os.pipe()
        spec["emit_fd"] = w_fd
        spec_path = out_dir / f"spec_{tree_name}.json"
        spec_path.write_text(json.dumps(spec, indent=1))
        if run is not None:
            run.heartbeat(stage=f"starting the {tree_name} worker ({len(tree_units)} units)")
        child = subprocess.Popen(
            [str(PYTHON), "-B", str(HERE / "worker.py"), str(spec_path)],
            cwd=str(WIDER),
            env=worker_env(tree_name),
            stdin=subprocess.DEVNULL,
            pass_fds=(w_fd,),
        )
        current["child"] = child
        os.close(w_fd)
        print(f"# worker {tree_name} pid {child.pid}", flush=True)
        expected = {u["uid"] for u in tree_units}
        got = set()
        with os.fdopen(r_fd, "r") as reader:
            for line in reader:
                msg = json.loads(line)
                kind = msg.pop("kind")
                if kind == "unit":
                    data = msg["data"]
                    data["worker_tree"] = tree_name
                    uid = data["unit"]
                    if uid not in expected or uid in got:
                        problems.append(f"unexpected unit {uid} from the {tree_name} worker")
                    got.add(uid)
                    seen.append(uid)
                    if data["tree_head"] != TREES[tree_name][1] or data["tree_status_short"].strip():
                        problems.append(f"{uid}: tree {data['tree_head']} status {data['tree_status_short']!r}")
                    if data["status"] == "error":
                        problems.append(f"{uid}: error")
                    if data.get("probe_error"):  # prompt 05a
                        problems.append(f"{uid}: probe error")
                    if tree_name == "t5":  # prompt 06: the baseline tree
                        p_wall[data["row"]] = data["wall_s"]
                        p_status[data["row"]] = data["status"]
                    record(uid, data)
                else:
                    (out_dir / f"worker_{tree_name}_{kind}.json").write_text(json.dumps(msg, indent=1))
                    if kind == "refused":
                        problems.append(f"the {tree_name} worker refused: {msg}")
        rc = child.wait()
        current["child"] = None
        print(f"# worker {tree_name} exit {rc}", flush=True)
        if rc != 0:
            problems.append(f"the {tree_name} worker exited {rc}")
        missing = expected - got
        if missing:
            problems.append(f"the {tree_name} worker did not report {sorted(missing)}")

    try:
        # prompt 06: t5 is the baseline (05a's c), s is measured against it (05a's t)
        p_units = [u for u in units if u["tree"] == "t5"]
        run_worker("t5", p_units)
        c_units = [u for u in units if u["tree"] == "s"]
        for u in c_units:
            if args.budget_override is not None:
                u["budget"] = args.budget_override
            elif p_status.get(u["row"]) == "done" and p_wall.get(u["row"]) is not None:
                u["budget"] = BUDGET_FACTOR * p_wall[u["row"]]
            else:
                u["budget"] = None
                u["budget_note"] = f"no budget: the t5 unit of {u['row']} is {p_status.get(u['row'])}"
        run_worker("s", c_units)
    except BaseException:
        traceback.print_exc()
        if run is not None:
            run.finish("failed", exit_code=1)
        raise

    order_ok = seen == [u["uid"] for u in units]
    if not order_ok:
        problems.append("the units were not recorded in the planned order")
    print(f"\n== run ends: {len(seen)} of {len(units)} units recorded; in order {order_ok}; "
          f"problems {problems}", flush=True)
    ok = len(problems) == 0
    if run is not None:
        run.finish("done" if ok else "failed", exit_code=0 if ok else 2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
