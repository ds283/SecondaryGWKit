"""prompt 07b extension (section 2.7): Part A's scripts/run_replays_07b.py copied, changed only where marked "prompt 07b
extension": replay_ext.py, outputs under out/ext/ (replays/, replay_logs/), and --instrument passed on. Its text follows.

levin-phase-floor prompt 07b Part A: run replay_07b.py on many calls, each in its own process, at most --workers at a
time, and summarise. New in 07b (after 07a Part A's run_replays_07a.py). Each replay's stdout and stderr go to
out/replay_logs/<variant>/<sid>.log. Writes out/replays/<variant>_summary_<label>.json (refuses to overwrite).
Usage: PYTHONPATH=<tree> python -B run_replays_07b.py VARIANT LABEL (--sets S [S ...] | --sids SID [SID ...]) [--workers 6]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # prompt 07b extension: scripts/ext/
ROOT = HERE.parent.parent
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("label")
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("--sids", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--instrument", action="store_true")  # prompt 07b extension
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit("at most six workers")
    if os.environ.get("PYTHONPATH", "").split(":")[0] != TREE:
        raise SystemExit("PYTHONPATH must start with the tree")
    man = json.loads((ROOT / "state" / "calls_07b.json").read_text())["calls"]
    sids = a.sids if a.sids else sorted(s for s, c in man.items() if c["set"] in a.sets)
    out_s = ROOT / "out" / "ext" / "replays" / f"{a.variant}_summary_{a.label}.json"  # prompt 07b extension
    if out_s.exists():
        raise SystemExit(f"{out_s} exists")
    logd = ROOT / "out" / "ext" / "replay_logs" / a.variant  # prompt 07b extension
    logd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=TREE, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    pending, running, rc = list(sids), {}, {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.workers:
            s = pending.pop(0)
            log = open(logd / f"{s}.log", "w")
            running[s] = subprocess.Popen([PY, "-B", str(HERE / "replay_ext.py"), s, a.variant, str(ROOT / "out" / "ext" / "replays")] + (["--instrument"] if a.instrument else []),
                                          cwd=TREE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
        for s, p in list(running.items()):
            if p.poll() is not None:
                rc[s] = p.returncode
                del running[s]
        time.sleep(0.2)
    rows = {}
    for s in sids:
        p = ROOT / "out" / "ext" / "replays" / a.variant / f"{s}.json"  # prompt 07b extension
        if p.exists():
            d = json.loads(p.read_text())
            rows[s] = {"rc": rc[s], "reproduces_census": d["reproduces_census"], "problems": d["problems"], "replay": d["replay"],
                       "census": d["census"], "wall_s": d["wall_s"], "n_windows": d["n_windows"]}
        else:
            rows[s] = {"rc": rc[s], "reproduces_census": None, "problems": ["no output"]}
    n_ok = sum(1 for r in rows.values() if r["reproduces_census"])
    out_s.parent.mkdir(parents=True, exist_ok=True)
    out_s.write_text(json.dumps({"variant": a.variant, "label": a.label, "n": len(sids), "n_reproduce": n_ok,
                                 "wall_s": time.time() - t0, "rows": rows}, indent=1))
    for s, r in rows.items():
        if not r["reproduces_census"]:
            print(f"  {s}: rc {r['rc']} {r['problems']}")
    print(f"{a.variant} {a.label}: {n_ok} of {len(sids)} reproduce their census record; {time.time() - t0:.0f} s; wrote {out_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
