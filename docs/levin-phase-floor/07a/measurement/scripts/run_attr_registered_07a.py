"""levin-phase-floor prompt 07a Part A, section 2.2: the attribution's leaf references, as a registered run (CLAUDE.md,
"Long-running jobs"). New in 07a.

For each sid in order: ``attr_07a.py SID out/replays/a/SID.json out/attr/SID.json --res 1 2 --workers W`` in its own
process (skipped if its output exists), stdout and stderr to out/attr/SID.log. The runregistrykit run RUN_DIR is
heartbeated at start (this pid), after each call and every 60 s, and finished ``done`` (every call ok), ``failed``
or ``killed`` (SIGTERM, the child killed). Usage: PYTHONPATH=<tree>:<this dir> python -B run_attr_registered_07a.py
RUN_DIR W SID [SID ...]
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import runregistrykit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    run_dir, workers, sids = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    if workers > 6:
        raise SystemExit("at most six worker processes")
    run = runregistrykit.Run(run_dir)
    run.heartbeat(pid=os.getpid(), units_done=0, units_total=len(sids), stage="started")
    done = []
    child = {}
    stop = threading.Event()

    def beat():
        while not stop.wait(60):
            try:
                run.heartbeat(units_done=len(done), units_total=len(sids))
            except Exception:
                pass

    threading.Thread(target=beat, daemon=True).start()

    def on_term(signum, frame):
        p = child.get("p")
        if p is not None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
        run.finish("killed", exit_code=128 + signum)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    results = []
    for sid in sids:
        out = ROOT / "out" / "attr" / f"{sid}.json"
        if out.exists():
            results.append({"sid": sid, "status": "skipped (exists)"})
            done.append(sid)
            continue
        t0 = time.time()
        with open(ROOT / "out" / "attr" / f"{sid}.log", "wb") as log:
            p = subprocess.Popen([PY, "-B", str(HERE / "attr_07a.py"), sid, str(ROOT / "out" / "replays" / "a" / f"{sid}.json"),
                                  str(out), "--res", "1", "2", "--workers", str(workers)], env=env, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            child["p"] = p
            rc = p.wait()
            child["p"] = None
        results.append({"sid": sid, "status": "ok" if rc == 0 else f"exit {rc}", "wall_s": time.time() - t0})
        done.append(sid)
        print(json.dumps(results[-1]), flush=True)
        run.heartbeat(units_done=len(done), units_total=len(sids), stage=f"{sid} {results[-1]['status']}")
    stop.set()
    bad = [r for r in results if r["status"] not in ("ok", "skipped (exists)")]
    (ROOT / "out" / "attr" / f"done_{Path(run_dir).name}.json").write_text(json.dumps(
        {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "workers": workers, "results": results}, indent=1))
    run.finish("done" if not bad else "failed", exit_code=0 if not bad else 1)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
