"""
levin-phase-floor prompt 06e section 5: the registered run (slug ``census``), 06c's census re-run at
06e's code. Copied from prompt 06c's census scripts/run_census.py
(docs/levin-phase-floor/06c/census/scripts/run_census.py, a record that is not edited; its copy is
kept in state/scripts_06c_reference/). Unchanged in its code: the tree ``g`` (the default) and
``p`` are prov.TREES, which prompt 06e moves to the worktrees at 6d81fad and 50277f3; every unit
is ``<tree>-<rtol>-<shard>-<serial>``, every 1e-7 unit carries ``"w3"`` and ``"scan"``, and no
unit carries ``"capture"``, as in 06c. Changed (prompt 06e) only in this paragraph.
06c's text follows, unchanged.

levin-phase-floor prompt 06c section 5: the registered run (slug ``census``), 06b's census re-run
under the global error budget. Copied from prompt 06b's census scripts/run_census.py
(docs/levin-phase-floor/06b/census/scripts/run_census.py, a record that is not edited; its copy is
kept in state/scripts_06b_reference/). Changed here, each change marked "prompt 06c":
- **the tree** is ``g`` (prov.TREES: the worktree at Part B's dispatch SHA e5764a6, whose code is
  06c's plus 06d's) by default, ``--tree p`` for a run at the parent; the units are
  ``<tree>-<rtol>-<shard>-<serial>``;
- **no capture list**: 06b's W2 capture (state/w2_capture.json) is not read, and no unit of the
  run carries ``"capture"``. X1's 24 calls are known only after the run, and their inputs are
  captured then, by re-running their units at g (x1_capture.py); every 1e-7 unit still carries
  ``"w3": true`` and ``"scan": true``, as in 06b.
06b's text follows, unchanged; read its ``b`` as ``g``.

levin-phase-floor prompt 06b section 5: the registered run (slug ``census``), 06a's census re-run
at the code commit. Copied from prompt 06a's scripts/run_census.py
(docs/levin-phase-floor/06a/scripts/run_census.py, a record that is not edited; its copy is kept in
state/scripts_06a_reference/). Changed here, each change marked "prompt 06b":
- **the tree** is ``b`` (prov.TREES: the worktree at Part B's dispatch SHA, whose code is the code
  commit's) by default, ``--tree p`` for a run at the parent; the units are
  ``<tree>-<rtol>-<shard>-<serial>``;
- **every 1e-7 unit** also carries ``"w3": true`` (W3's evidence, worker.py), and, if it is listed in
  state/w2_capture.json (06a's phase-limited 1e-7 calls with no sign change of theta' in 06a's scan;
  make_capture.py), ``"capture"``: the [call index, notify_label] pairs whose inputs the worker
  pickles for W2's reference. The file's SHA-256 is in the header and in every spec.
06a's text follows, unchanged; read its ``h`` as the tree.

levin-phase-floor prompt 06a: the registered run (slug ``census``). The driver. Copied from
prompt 06's scripts/run_wider.py (docs/levin-phase-floor/06/wider/scripts/run_wider.py, a record
that is not edited; prompt 06a section 2 item 4: "run_wider.py becomes run_census.py"), and
rewritten where the census differs:

- **One tree,** ``h`` (prov.TREES), the worktree at the dispatch SHA; no baseline set, no budget
  taken from another tree.
- **The units** (section 2 item 5): one per (rtol, row), ``h-1e-7-<shard>-<serial>`` on all 9,280
  rows of state/rows.json, then ``h-1e-8-<shard>-<serial>`` on all 9,280. ``atol`` is A3 v2's.
  Every 1e-7 unit carries the stationary-point scan (``"scan": true``).
- **The workers** (section 2 item 4): one worker per k per rtol (rows.json's ``k_key``), its
  units in rows.json's order (block by block in z_response), at most six running at once
  (``--max-workers``, default 6). All of the k workers for ``1e-7`` finish before the ``1e-8``
  workers start. The workers are started longest first (most units first; all eight have 1,160
  here). Each worker re-runs its own k's producers once (rebuild.CensusCache).
- **The budget** (section 2 item 6): a fixed 300 s per unit, raised from the dispatcher as in
  04-06 (``BUDGET_S``).
- The driver reads every worker's pipe (one reader thread per worker, into one queue that the main
  thread drains), and records one checkpoint per unit (``Run.record``), as 06's driver did. A
  worker's ``event`` lines (what its producer cache built) go to ``events_<worker>.jsonl`` in the
  run directory, its other messages to ``worker_<worker>_<kind>.json``.
- Heartbeat thread every 60 s; on SIGTERM or SIGINT the driver records ``killed``, sends SIGTERM to
  every worker it is running, and dies by the signal.
- ``--skip-recorded-in RUN_DIR``: a resume's driver (section 6: "the resume is a new run that skips
  units already recorded") skips every unit that RUN_DIR's checkpoint holds; nothing is copied or
  re-stamped.
- **The order check** is per worker: each worker's units must be recorded in its planned order;
  across workers they interleave.

06's run_wider.py is kept, unchanged, in state/scripts_06_reference/; the diff against it is
saved at launch and at the end.

Usage: python -B run_census.py --run-dir RUN_DIR      (the registered run; launch.py starts it)
       python -B run_census.py --out-dir DIR [--rows N1,N2] [--k-keys K1] [--rtols 1e-7]
                                                       (C0, the pilot, a smoke test)
"""

import argparse
import datetime
import json
import os
import queue
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

BUDGET_S = 300.0
SCRIPTS = [
    "run_census.py",
    "worker.py",
    "dispatch.py",
    "harness.py",
    "rebuild.py",
    "prov.py",
    "launch.py",
    "rerun.py",
    "spscan.py",
]
RTOLS = ("1e-7", "1e-8")


def plan(rows_doc, names=None, rtols=RTOLS, k_keys=None, skip=(), tree="g", capture=None):  # prompt 06c
    """Worker specs, per rtol, in launch order: {rtol: [(worker_name, [unit, ...]), ...]}.
    prompt 06b: ``tree`` names the tree; ``capture`` maps "<rtol>-<shard>-<serial>" to the
    [index, notify_label] pairs of W2's inputs."""
    capture = capture or {}
    rows = rows_doc["rows"]
    if names is not None:
        names = set(names)
        rows = [r for r in rows if r["name"] in names or (r["names05"] and set(r["names05"]) & names)]
    if k_keys is not None:
        rows = [r for r in rows if r["k_key"] in k_keys]
    out = {}
    for rtol in rtols:
        by_k = {}
        for r in rows:
            uid = f"{tree}-{rtol}-{r['shard']}-{r['serial']}"  # prompt 06b
            if uid in skip:
                continue
            by_k.setdefault(r["k_key"], []).append(
                {
                    "uid": uid,
                    "tree": tree,  # prompt 06b
                    "rtol": rtol,
                    "row": r["name"],
                    "names05": r["names05"],
                    "shard": r["shard"],
                    "serial": r["serial"],
                    "k_key": r["k_key"],
                    "budget": BUDGET_S,
                    "scan": rtol == "1e-7",
                    "w3": rtol == "1e-7",  # prompt 06b
                    "capture": capture.get(f"{rtol}-{r['shard']}-{r['serial']}"),  # prompt 06b
                }
            )
        workers = sorted(by_k.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        out[rtol] = [
            (f"{rtol}-k{kk.replace(':', '_')}", units) for kk, units in workers
        ]
    return out


def worker_env(tree="g"):  # prompt 06b; prompt 06c: g
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TREES[tree][0])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-dir")
    g.add_argument("--run-dir")
    parser.add_argument("--rows", default=None, help="smoke test only: row names, comma-separated")
    parser.add_argument("--k-keys", default=None, help="smoke test only")
    parser.add_argument("--rtols", default="1e-7,1e-8")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--skip-recorded-in", default=None)
    parser.add_argument("--tree", default="g", choices=sorted(TREES))  # prompt 06b; prompt 06c: g
    args = parser.parse_args()
    if args.max_workers > 6:
        raise SystemExit("at most six workers (the Mac Studio's --cpus 6 rule)")

    rows_path = WIDER / "state" / "rows.json"
    rows_doc = json.loads(rows_path.read_text())
    rows_sha = sha256(rows_path)
    skip = set()
    if args.skip_recorded_in:
        with open(Path(args.skip_recorded_in) / "checkpoint.jsonl") as f:
            for line in f:
                skip.add(json.loads(line)["unit"])
    rtols = tuple(args.rtols.split(","))
    names = None if args.rows is None else args.rows.split(",")
    k_keys = None if args.k_keys is None else args.k_keys.split(",")
    # prompt 06b: W2's inputs to capture (state/w2_capture.json, make_capture.py)
    # prompt 06c: none; X1's inputs are captured after the run (x1_capture.py)
    capture_sha = None
    plans = plan(rows_doc, names, rtols, k_keys, skip, tree=args.tree, capture=None)

    run = None
    children = {}
    if args.run_dir is not None:
        if args.rows or args.k_keys or args.rtols != "1e-7,1e-8" or args.tree != "g":  # prompt 06b; 06c: g
            raise SystemExit("--rows/--k-keys/--rtols are for the smoke test only")
        import runregistrykit

        run = runregistrykit.Run(args.run_dir)
        out_dir = Path(run.path)

        def terminal(signum, _frame):
            for child in list(children.values()):
                if child.poll() is None:
                    try:
                        child.send_signal(signal.SIGTERM)
                    except Exception:
                        pass
            run.finish("killed", exit_code=128 + signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminal)
        run.heartbeat(pid=os.getpid(), stage="started: the 1e-7 workers are next")

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
    all_units = [u["uid"] for rt in rtols for _, us in plans[rt] for u in us]
    header = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_hashes["run_census.py"],
        "scripts_sha256": script_hashes,
        "rows_sha256": rows_sha,
        "w2_capture_sha256": capture_sha,  # prompt 06b
        "tree": args.tree,  # prompt 06b
        "trees": {k: [str(v[0]), v[1]] for k, v in TREES.items()},
        "host": socket.gethostname(),
        "python": sys.version,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_units": len(all_units),
        "workers": {rt: [(w, len(us)) for w, us in plans[rt]] for rt in rtols},
        "budget_s": BUDGET_S,
        "max_workers": args.max_workers,
        "skipped_as_recorded": len(skip),
    }
    print("# run_census header " + json.dumps(header, default=str), flush=True)

    problems = []
    seen = []
    n_recorded = [0]

    def record(uid, data):
        if run is not None:
            run.record(uid, data)
            n_recorded[0] += 1
            if n_recorded[0] % 10 == 0 or data["status"] != "done":
                run.heartbeat(stage=f"{n_recorded[0]} units recorded; last {uid} {data['status']}")
        else:
            with open(out_dir / "ledger.jsonl", "a") as f:
                f.write(json.dumps({"unit": uid, "data": data}) + "\n")

    def run_set(rtol, workers):
        """Run one rtol's workers, at most max_workers at once."""
        q = queue.Queue()
        pending = list(workers)
        running = {}  # name -> (child, expected list, got list)

        def reader(name, r_fd):
            with os.fdopen(r_fd, "r") as rd:
                for line in rd:
                    q.put((name, line))
            q.put((name, None))

        def start(name, units):
            spec = {
                "tree": args.tree,  # prompt 06b
                "name": name,
                "units": units,
                "out_dir": str(out_dir),
                "emit_fd": None,
                "emit_file": None,
                "rows_sha256": rows_sha,
                "w2_capture_sha256": capture_sha,  # prompt 06b
            }
            r_fd, w_fd = os.pipe()
            spec["emit_fd"] = w_fd
            spec_path = out_dir / f"spec_{name}.json"
            spec_path.write_text(json.dumps(spec, indent=1))
            log_path = out_dir / f"worker_{name}.log"
            log = open(log_path, "ab")
            child = subprocess.Popen(
                [str(PYTHON), "-B", str(HERE / "worker.py"), str(spec_path)],
                cwd=str(WIDER),
                env=worker_env(args.tree),  # prompt 06b
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(w_fd,),
            )
            log.close()
            os.close(w_fd)
            children[name] = child
            running[name] = (child, [u["uid"] for u in units], [])
            threading.Thread(target=reader, args=(name, r_fd), daemon=True).start()
            print(f"# worker {name} pid {child.pid} ({len(units)} units)", flush=True)
            if run is not None:
                run.heartbeat(stage=f"started worker {name} ({len(units)} units)")

        while pending and len(running) < args.max_workers:
            start(*pending.pop(0))
        while running:
            name, line = q.get()
            child, expected, got = running[name]
            if line is None:
                rc = child.wait()
                print(f"# worker {name} exit {rc}", flush=True)
                if rc != 0:
                    problems.append(f"the {name} worker exited {rc}")
                missing = [u for u in expected if u not in got]
                if missing:
                    problems.append(f"the {name} worker did not report {len(missing)} units, first {missing[:5]}")
                if got != [u for u in expected if u in got]:
                    problems.append(f"the {name} worker's units were not recorded in its planned order")
                del running[name]
                children.pop(name, None)
                if pending:
                    start(*pending.pop(0))
                continue
            msg = json.loads(line)
            kind = msg.pop("kind")
            if kind == "unit":
                data = msg["data"]
                data["worker_tree"] = args.tree  # prompt 06b
                uid = data["unit"]
                if uid not in expected or uid in got:
                    problems.append(f"unexpected unit {uid} from the {name} worker")
                got.append(uid)
                seen.append(uid)
                if data["tree_head"] != TREES[args.tree][1] or data["tree_status_short"].strip():  # prompt 06b
                    problems.append(f"{uid}: tree {data['tree_head']} status {data['tree_status_short']!r}")
                if data["status"] == "error":
                    problems.append(f"{uid}: error")
                record(uid, data)
            elif kind == "event":
                with open(out_dir / f"events_{name}.jsonl", "a") as f:
                    f.write(json.dumps(msg) + "\n")
            else:
                (out_dir / f"worker_{name}_{kind}.json").write_text(json.dumps(msg, indent=1))
                if kind == "refused":
                    problems.append(f"the {name} worker refused: {msg}")

    try:
        for rtol in rtols:
            if run is not None:
                run.heartbeat(stage=f"rtol {rtol}: {len(plans[rtol])} workers")
            run_set(rtol, plans[rtol])
    except BaseException:
        traceback.print_exc()
        if run is not None:
            run.finish("failed", exit_code=1)
        raise

    n_err = sum(1 for p in problems if p.endswith(": error"))
    print(f"\n== run ends: {len(seen)} of {len(all_units)} units recorded; "
          f"units in error {n_err}; problems {problems[:50]}", flush=True)
    ok = len(problems) == 0 and len(seen) == len(all_units)
    if run is not None:
        run.finish("done" if ok else "failed", exit_code=0 if ok else 2)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
