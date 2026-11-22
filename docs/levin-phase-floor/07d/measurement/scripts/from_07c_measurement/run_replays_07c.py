"""levin-phase-floor prompt 07c Part A: 07b's extension scripts/ext/run_replays_ext.py (a record) copied, changed only
where marked "prompt 07c": it lives in scripts/ (ROOT one level up), runs replay_07c.py at this worktree (2fd979fd), reads
the manifests state/calls_07c.json and state/calls_07c_r5.json (if present), and writes under out/replays/ and
out/replay_logs/. 07b's extension text follows.

prompt 07b extension (section 2.7): Part A's scripts/run_replays_07b.py copied, changed only where marked "prompt 07b
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

HERE = Path(__file__).resolve().parent  # prompt 07c: scripts/
ROOT = HERE.parent  # prompt 07c
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f"  # prompt 07c
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
    man = json.loads((ROOT / "state" / "calls_07c.json").read_text())["calls"]  # prompt 07c
    if (ROOT / "state" / "calls_07c_r5.json").exists():  # prompt 07c
        man.update(json.loads((ROOT / "state" / "calls_07c_r5.json").read_text())["calls"])
    sids = a.sids if a.sids else sorted(s for s, c in man.items() if c["set"] in a.sets)
    out_s = ROOT / "out" / "replays" / f"{a.variant}_summary_{a.label}.json"  # prompt 07c
    if out_s.exists():
        raise SystemExit(f"{out_s} exists")
    logd = ROOT / "out" / "replay_logs" / a.variant  # prompt 07c
    logd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=TREE, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    pending, running, rc = list(sids), {}, {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.workers:
            s = pending.pop(0)
            log = open(logd / f"{s}.log", "w")
            running[s] = subprocess.Popen([PY, "-B", str(HERE / "replay_07c.py"), s, a.variant, str(ROOT / "out" / "replays")] + (["--instrument"] if a.instrument else []),
                                          cwd=TREE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
        for s, p in list(running.items()):
            if p.poll() is not None:
                rc[s] = p.returncode
                del running[s]
        time.sleep(0.2)
    rows = {}
    for s in sids:
        p = ROOT / "out" / "replays" / a.variant / f"{s}.json"  # prompt 07c
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
