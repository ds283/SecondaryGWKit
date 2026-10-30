"""depth20-tiling: run runcall.py for the sampled calls in one mode, each in its own fresh process,
at most six at once, each with a wall-time limit (killed and reported past it).

Usage: PYTHONPATH=<tree>:<scripts> python -B run_calls.py CAPTURED_JSON MODE OUT_DIR [--workers 6] [--timeout 1800] [--only 0,1,...]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import header  # noqa: E402

PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captured")
    ap.add_argument("mode")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit("at most six workers")
    for l in header(Path(__file__).resolve(), f"depth20-tiling: runcall {a.mode} over the sample", [Path(a.captured), HERE / "runcall.py", HERE / "dcommon.py"]):
        print(l, flush=True)
    ents = json.loads(Path(a.captured).read_text())
    nos = [e["sample_no"] for e in ents]
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        nos = [n for n in nos if n in keep]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pending = list(nos)
    running = {}
    status = {}
    while pending or running:
        while pending and len(running) < a.workers:
            no = pending.pop(0)
            log = open(out_dir / f"{no:02d}_{a.mode}.log", "w")
            p = subprocess.Popen([PY, "-B", str(HERE / "runcall.py"), a.captured, str(no), a.mode, str(out_dir / f"{no:02d}_{a.mode}.json")],
                                 env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            running[no] = (p, time.time())
            print(f"sample {no}: pid {p.pid}", flush=True)
        for no, (p, t0) in list(running.items()):
            if p.poll() is not None:
                status[no] = {"rc": p.returncode, "wall": time.time() - t0}
                del running[no]
                print(f"sample {no}: exit {p.returncode} after {time.time() - t0:.1f} s", flush=True)
            elif time.time() - t0 > a.timeout:
                p.kill()
                p.wait()
                status[no] = {"rc": "killed at timeout", "wall": time.time() - t0}
                del running[no]
                print(f"sample {no}: KILLED at the {a.timeout:.0f} s limit", flush=True)
        time.sleep(0.2)
    print(json.dumps(status, indent=1))
    bad = [n for n, s in status.items() if s["rc"] != 0]
    print(f"done; non-zero or killed: {bad}")


if __name__ == "__main__":
    main()
