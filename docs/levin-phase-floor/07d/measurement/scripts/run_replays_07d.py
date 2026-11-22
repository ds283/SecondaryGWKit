"""levin-phase-floor prompt 07d Part A: run replay_07d.py over a list of sids for one or more codes, at most WORKERS (<= 6)
processes at once, each given a share of the sids balanced by the calls' census ``elapsed`` (longest first, to the least
loaded share). New in 07d. Each process's output goes to OUT_DIR/logs/<variant>-<k>.log; a summary line per code.
Usage: PYTHONPATH=<tree> python -B run_replays_07d.py OUT_DIR SIDS_FILE VARIANT [VARIANT ...] [--workers N]
"""

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def main():
    args = sys.argv[1:]
    workers = 6
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    if workers > 6:
        raise SystemExit("at most six workers")
    out_dir, sids_file, variants = Path(args[0]), Path(args[1]), args[2:]
    sids = [s for s in sids_file.read_text().split() if s]
    man = {}
    for p in sorted((ROOT / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    el = {r["sid"]: r["elapsed"] or 1.0 for r in pickle.loads((ROOT / "state" / "census_index_07d.pkl").read_bytes())["rows"]}
    cost = {s: el.get(man[s]["census"], 5.0) + 0.5 for s in sids}
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    jobs = []
    for v in variants:
        shares = [[0.0, []] for _ in range(workers)]
        for s in sorted(sids, key=lambda s: -cost[s]):
            sh = min(shares, key=lambda x: x[0])
            sh[0] += cost[s]
            sh[1].append(s)
        for k, (_, ss) in enumerate(shares):
            if ss:
                f = out_dir / "logs" / f"{v}-{k}.sids"
                f.write_text("\n".join(ss) + "\n")
                jobs.append((v, k, f))
    env = dict(os.environ, PYTHONPATH=TREE, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    running, rc = {}, {}
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < workers:
            v, k, f = jobs.pop(0)
            log = open(out_dir / "logs" / f"{v}-{k}.log", "w")
            running[(v, k)] = subprocess.Popen([PY, "-B", str(HERE / "replay_07d.py"), v, str(out_dir), "--sids-file", str(f)],
                                               cwd=TREE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
        for key, p in list(running.items()):
            if p.poll() is not None:
                rc[key] = p.returncode
                del running[key]
        time.sleep(0.5)
    for v in variants:
        n = rep = 0
        for f in sorted((out_dir / v).glob("*.json")):
            d = json.loads(f.read_text())
            if d["sid"] in sids:
                n += 1
                rep += d["reproduces_census"]
        print(f"{v}: {n} of {len(sids)} written; {rep} reproduce their census record; exit codes "
              f"{ {k: c for (vv, k), c in rc.items() if vv == v} }")
    print(f"wall {time.time() - t0:.0f} s")
    return 1 if any(c != 0 for c in rc.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
