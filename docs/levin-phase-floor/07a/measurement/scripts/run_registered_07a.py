"""levin-phase-floor prompt 07a Part A, the registered run (CLAUDE.md, "Long-running jobs"): run_replays_07a.py with
the runregistrykit run RUN_DIR heartbeated (at start with this pid, after every job, and every 60 s by a thread)
and finished ``done`` (every job ok), ``failed`` (any not ok) or ``killed`` (SIGTERM). Otherwise as
run_replays_07a.py, whose text follows.

levin-phase-floor prompt 07a Part A: run replay_07a.py over a list of (call, variant), each in its own process, at most
--procs at once (the brief: at most six worker processes). New in 07a.

The plan is a JSON list of [sid, variant, instrument]. A job whose output exists is skipped (so a run can be
resumed); each job's stdout and stderr go to LOG_DIR/<variant>/<sid>.log; each has a wall-clock limit (--timeout
seconds, default none), and a job over it is killed and recorded as ``timeout`` (its call is then reported, never
silently dropped). Writes DONE_JSON at the end: every job's exit status and wall time, with this script's SHA-256.
With --heartbeat RUN_DIR the runregistrykit run there is heartbeated after each job.

Usage: PYTHONPATH=<tree> python -B run_replays_07a.py PLAN_JSON OUT_DIR LOG_DIR DONE_JSON [--procs 6] [--timeout S]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out_dir")
    ap.add_argument("log_dir")
    ap.add_argument("done_json")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--heartbeat", required=True, help="the run directory")
    a = ap.parse_args()
    if a.procs > 6:
        raise SystemExit("at most six worker processes")
    plan = json.loads(Path(a.plan).read_text())
    out_dir, log_dir = Path(a.out_dir), Path(a.log_dir)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    import signal
    import threading

    import runregistrykit

    run = runregistrykit.Run(a.heartbeat)
    run.heartbeat(pid=os.getpid(), units_done=0, units_total=len(plan), stage="started")
    stop_beat = threading.Event()
    procs_alive = {}

    def beat():
        while not stop_beat.wait(60):
            try:
                run.heartbeat(units_done=len(results), units_total=len(plan))
            except Exception:
                pass

    threading.Thread(target=beat, daemon=True).start()

    def on_term(signum, frame):
        for p_ in list(procs_alive.values()):
            try:
                p_.kill()
            except Exception:
                pass
        run.finish("killed", exit_code=128 + signum)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    results = []
    t_start = time.time()

    def job(item):
        sid, variant, instrument = item
        out = out_dir / variant / f"{sid}.json"
        if out.exists():
            return {"sid": sid, "variant": variant, "status": "skipped (exists)"}
        lp = log_dir / variant / f"{sid}.log"
        lp.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-B", str(HERE / "replay_07a.py"), sid, variant, str(out_dir)] + (["--instrument"] if instrument else [])
        t0 = time.time()
        with open(lp, "wb") as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            procs_alive[(sid, variant)] = p
            try:
                rc = p.wait(timeout=a.timeout)
                status = "ok" if rc == 0 else f"exit {rc}"
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                status = "timeout"
            procs_alive.pop((sid, variant), None)
        return {"sid": sid, "variant": variant, "status": status, "wall_s": time.time() - t0}

    with ThreadPoolExecutor(max_workers=a.procs) as ex:
        for r in ex.map(job, plan):
            results.append(r)
            print(json.dumps(r), flush=True)
            try:
                run.heartbeat(units_done=len(results), units_total=len(plan), stage=f"{r['sid']} {r['variant']} {r['status']}")
            except Exception:
                pass
    doc = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "plan": str(Path(a.plan).resolve()),
           "plan_sha256": sha256(a.plan), "wall_s": time.time() - t_start, "results": results}
    Path(a.done_json).write_text(json.dumps(doc, indent=1))
    bad = [r for r in results if r["status"] not in ("ok", "skipped (exists)")]
    print(f"{len(results)} jobs, {len(bad)} not ok")
    stop_beat.set()
    run.finish("done" if not bad else "failed", exit_code=0 if not bad else 1)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
