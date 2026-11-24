"""levin-phase-floor prompt 07d Part A': run replay_a2.py over a list of sids for one or more codes, at most WORKERS (<= 6)
processes at once, each given a share of the sids balanced by Part A's tree replay walls (longest first, to the least
loaded share). After Part A's run_replays_07d.py (a record), changed in its replay script, its tree and its balance.
Each process's output goes to OUT_DIR/logs/<variant>-<k>.log; a summary line per code.
Usage: python -B run_replays_a2.py OUT_DIR SIDS_FILE VARIANT [VARIANT ...] [--workers N]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PARTA, PY, TREE  # noqa: E402


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


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
    cost = {}
    for s in sids:
        d = json.loads((PARTA / "out" / "replays" / "tree" / f"{fname(s)}.json").read_text())
        cost[s] = float(d.get("wall_s") or 1.0) + 0.05
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
    env = dict(os.environ, PYTHONPATH=f"{TREE}:{HERE}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    running, rc = {}, {}
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < workers:
            v, k, f = jobs.pop(0)
            log = open(out_dir / "logs" / f"{v}-{k}.log", "w")
            running[(v, k)] = subprocess.Popen([PY, "-B", str(HERE / "replay_a2.py"), v, str(out_dir), "--sids-file", str(f)],
                                               cwd=str(TREE), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT)
            log.close()
        for key, p in list(running.items()):
            if p.poll() is not None:
                rc[key] = p.returncode
                del running[key]
        time.sleep(0.5)
    for v in variants:
        n = eq = 0
        for f in sorted((out_dir / v).glob("*.json")):
            d = json.loads(f.read_text())
            if d["sid"] in sids:
                n += 1
                eq += d["equals_tree"]
        print(f"{v}: {n} of {len(sids)} written; {eq} equal Part A's tree replay; exit codes "
              f"{ {k: c for (vv, k), c in rc.items() if vv == v} }")
    print(f"wall {time.time() - t0:.0f} s")
    return 1 if any(c != 0 for c in rc.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
