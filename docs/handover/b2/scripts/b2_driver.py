"""
handover prompt 13 (B2), Part B: the registered run (campaign handover, prompt 13, slug residue).
The driver. `b2_launch.py` starts it, detached. Adapted from A4's `a4_driver.py` (a record that
is not edited).

It runs prompt 13 section 2.4's 208 cells -- A2's ladder (`b2_cells.LADDER`, 16 rungs) in the 13
objects (`b2_cells.OBJECTS`: E, R, E', five add-one and five remove-one hybrids) -- each in a
child process of its own (`b2_cells.py --cell ...`), at most `PARALLEL` = 2 at once (the laptop
carries other projects' work), each with a wall-clock budget of `BUDGET_S` = 1 hour.

**Order.** Cheapest first: every E cell (the exact flavour, rung by rung upward across the
shapes), then rung by rung upward (980, 1e4, ..., 1e8), within a rung the shapes in A2's order and
the objects E', R, E'+i, R-i. **A raise stops that shape's next rung** (prompt 13 section 6): a
cell depends on every cell of the same shape at a strictly lower rung that is planned before it,
starts only once they have all ended, and is recorded `skipped`, never started, if any of them
`raised` or was itself skipped. (So an E cell depends only on the E cells below it; every other
cell on every cell of its shape below its rung.) A cell over its budget is killed (SIGKILL to its
process group) and recorded `not reached`, with its elapsed time: a result, not a failure; the
rungs above it still run.

**Records.** One checkpoint record per cell (`Run.record`, which stamps this script's SHA-256, the
launch's `git_head` and dirty flag). The payload is the worker's record (`b2_cells.run_cell`) plus
the SHA-256 of every script here, the tree's `HEAD` and `git status --short` taken when the cell
ended (`git --no-optional-locks`, so that nothing in `.git` is written), the worker's exit code,
its wall time and the load average at its end. The worker's own JSON, stdout and stderr stay in
the run directory's `cells/`.

**Ending.** `done` (exit 0) if every cell is `ok` or `not reached`; otherwise `failed` (exit 2),
with the reasons in stdout and in `summary.json`. On SIGTERM or SIGINT the driver kills the
cells in flight, records `killed`, and dies by the signal. A heartbeat thread refreshes the
registry heartbeat every 60 s.

Usage: python -B b2_driver.py --run-dir RUN_DIR                  (the registered run)
       python -B b2_driver.py --out-dir DIR [--cells UID,...] [--budget S] [--parallel N]
                                                                  (a smoke test)

No Ray, no datastore, no lock. After landing, this file is a record of its own SHA-256.
"""

import argparse
import datetime
import hashlib
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

import b2_cells as C  # noqa: E402

REPO = C.REPO
WORKDIR = HERE.parent
PYTHON = Path(sys.executable)
PARALLEL = 2
BUDGET_S = 1.0 * 3600.0
POLL_S = 2.0
SCRIPTS = ("b2_driver.py", "b2_cells.py", "b2_checks.py", "b2_launch.py")
FINISHED = ("ok", "not reached")


def sha256(path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def uid_of(obj, shape, x_resp) -> str:
    return f"{obj}|{shape}|{x_resp:g}"


def file_stem(uid) -> str:
    return uid.replace("|", "__")


def plan():
    """The 208 cells in prompt 13 section 2.4's order, each with the cells it depends on."""
    rungs_all = sorted({x for rungs in C.LADDER.values() for x in rungs})
    cells = []
    for x in rungs_all:
        for shape, rungs in C.LADDER.items():
            if x in rungs:
                cells.append(("E", shape, x))
    for x in rungs_all:
        for shape, rungs in C.LADDER.items():
            if x in rungs:
                for obj in C.OBJECTS:
                    if obj != "E":
                        cells.append((obj, shape, x))
    units = []
    for obj, shape, x in cells:
        below = [
            u["uid"]
            for u in units
            if u["shape"] == shape and u["x_resp"] < x
        ]
        units.append(
            {
                "uid": uid_of(obj, shape, x),
                "object": obj,
                "shape": shape,
                "x_resp": x,
                "below": below,
            }
        )
    return units


def git_state():
    def run(args):
        try:
            return subprocess.run(
                ["git", "--no-optional-locks", *args],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return f"<git failed: {exc}>"

    return {
        "tree_head": run(["rev-parse", "HEAD"]).strip(),
        "tree_status_short": run(["status", "--short"]),
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir")
    group.add_argument("--out-dir")
    parser.add_argument("--cells", default=None, help="smoke test only")
    parser.add_argument("--budget", type=float, default=None, help="smoke test only")
    parser.add_argument("--parallel", type=int, default=None, help="smoke test only")
    args = parser.parse_args()

    units = plan()
    budget = BUDGET_S
    parallel = PARALLEL
    run = None
    running = {}

    if args.run_dir is not None:
        if args.cells or args.budget is not None or args.parallel is not None:
            raise SystemExit("--cells/--budget/--parallel are for the smoke test only")
        if len(units) != 208:
            raise SystemExit(f"planned {len(units)} cells, not 208")
        import runregistrykit

        run = runregistrykit.Run(args.run_dir)
        out_dir = Path(run.path)

        def terminal(signum, _frame):
            for proc in list(running.values()):
                try:
                    os.killpg(proc["popen"].pid, signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
            run.finish("killed", exit_code=128 + signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminal)
        run.heartbeat(pid=os.getpid(), stage="driver started")

        stop_beat = threading.Event()

        def beat():
            while not stop_beat.wait(60.0):
                try:
                    run.heartbeat()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=beat, daemon=True).start()
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.cells:
            wanted = args.cells.split(",")
            units = [u for u in units if u["uid"] in wanted]
            for u in units:
                u["below"] = [b for b in u["below"] if b in wanted]
        if args.budget is not None:
            budget = args.budget
        if args.parallel is not None:
            parallel = args.parallel

    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    script_hashes = {name: sha256(HERE / name) for name in SCRIPTS}
    header = {
        "script": str(Path(__file__).resolve()),
        "scripts_sha256": script_hashes,
        "realistic_large_x_sha256": C.A2.SCRIPT_SHA256,
        "b2_cells_sha256": C.SCRIPT_SHA256,
        "host": socket.gethostname(),
        "python": sys.version,
        "executable": str(PYTHON),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parallel": parallel,
        "budget_s": budget,
        "units": [u["uid"] for u in units],
        **git_state(),
    }
    print("# b2_driver header " + json.dumps(header, default=str), flush=True)

    status = {}
    order = []
    problems = []
    pending = list(units)

    def record(uid, data):
        status[uid] = data["status"]
        order.append(uid)
        if data["status"] not in FINISHED:
            problems.append(f"{uid}: {data['status']}: {data.get('error', '')}")
        if run is not None:
            run.record(uid, data)
            run.heartbeat(
                units_done=len(order),
                units_total=len(units),
                stage=f"cell {uid} {data['status']}",
            )
        else:
            with open(out_dir / "ledger.jsonl", "a") as handle:
                handle.write(json.dumps({"unit": uid, "data": data}, default=repr) + "\n")
        print(
            f"[{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] {uid}: "
            f"{data['status']}"
            + (
                f", {data.get('Levin_regions')} regions, depth {data.get('max_depth')}, "
                f"integral {data.get('integral_time', 0.0):.1f} s"
                if data["status"] == "ok"
                else f", {data.get('error', '')}"
            )
            + f" (wall {data.get('driver_wall_s', 0.0):.1f} s)",
            flush=True,
        )

    def provenance(data, unit, wall, rc):
        data.update(git_state())
        data["unit"] = unit["uid"]
        data["scripts_sha256"] = script_hashes
        data["driver_wall_s"] = wall
        data["worker_exit_code"] = rc
        data["loadavg_at_end"] = list(os.getloadavg())
        data["budget_s"] = budget
        data["ended"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return data

    def start(unit):
        stem = file_stem(unit["uid"])
        json_path = cells_dir / f"{stem}.json"
        argv = [
            str(PYTHON),
            "-B",
            str(HERE / "b2_cells.py"),
            "--cell",
            unit["shape"],
            repr(unit["x_resp"]),
            unit["object"],
            "--out",
            str(json_path),
        ]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        stdout = open(cells_dir / f"{stem}.out", "ab")
        stderr = open(cells_dir / f"{stem}.err", "ab")
        popen = subprocess.Popen(
            argv,
            cwd=str(WORKDIR),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )
        stdout.close()
        stderr.close()
        running[unit["uid"]] = {
            "unit": unit,
            "popen": popen,
            "t0": time.time(),
            "json": json_path,
        }
        print(f"# start {unit['uid']} pid {popen.pid}", flush=True)

    def finish_cell(uid, killed):
        entry = running.pop(uid)
        unit, popen = entry["unit"], entry["popen"]
        wall = time.time() - entry["t0"]
        if killed:
            try:
                os.killpg(popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = popen.wait()
            data = {
                "object": unit["object"],
                "shape": unit["shape"],
                "x_resp": unit["x_resp"],
                "status": "not reached",
                "elapsed_s": wall,
                "error": f"over the {budget:.0f} s budget; killed after {wall:.0f} s",
            }
        else:
            rc = popen.returncode
            try:
                data = json.loads(entry["json"].read_text())
            except Exception as exc:  # noqa: BLE001
                data = {
                    "object": unit["object"],
                    "shape": unit["shape"],
                    "x_resp": unit["x_resp"],
                    "status": "raised",
                    "error": f"the worker exited {rc} without a readable record ({exc})",
                }
            if data.get("status") == "ok" and rc != 0:
                data["status"] = "raised"
                data["error"] = f"the worker wrote ok but exited {rc}"
        record(uid, provenance(data, unit, wall, rc))

    try:
        while pending or running:
            # ended or over budget
            for uid in list(running):
                entry = running[uid]
                if entry["popen"].poll() is not None:
                    finish_cell(uid, killed=False)
                elif time.time() - entry["t0"] > budget:
                    finish_cell(uid, killed=True)
            # skip what a raise below has ruled out, start what is eligible
            progressed = True
            while progressed:
                progressed = False
                for unit in list(pending):
                    bad = [
                        b for b in unit["below"] if b in status and status[b] not in FINISHED
                    ]
                    if bad:
                        pending.remove(unit)
                        data = {
                            "object": unit["object"],
                            "shape": unit["shape"],
                            "x_resp": unit["x_resp"],
                            "status": "skipped",
                            "error": f"a cell below on this shape is not ok: {bad[0]} is {status[bad[0]]}",
                        }
                        record(unit["uid"], provenance(data, unit, 0.0, None))
                        progressed = True
                        break
                    if len(running) >= parallel:
                        continue
                    if all(status.get(b) in FINISHED for b in unit["below"]):
                        pending.remove(unit)
                        start(unit)
                        progressed = True
                        break
            if running:
                time.sleep(POLL_S)
    except BaseException:
        traceback.print_exc()
        for entry in running.values():
            try:
                os.killpg(entry["popen"].pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        if run is not None:
            run.finish("failed", exit_code=1)
        raise

    counts = {}
    for uid in order:
        counts[status[uid]] = counts.get(status[uid], 0) + 1
    if len(order) != len(units):
        problems.append(f"{len(order)} of {len(units)} cells recorded")
    summary = {
        "recorded": len(order),
        "planned": len(units),
        "counts": counts,
        "problems": problems,
        "order": order,
        "ended": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        f"\n== run ends: {len(order)} of {len(units)} cells recorded; {counts}; "
        f"problems {problems}",
        flush=True,
    )
    ok = not problems
    if run is not None:
        run.finish("done" if ok else "failed", exit_code=0 if ok else 2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
