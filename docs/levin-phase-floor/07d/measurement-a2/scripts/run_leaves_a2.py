"""levin-phase-floor prompt 07d Part A': run leaf_a2.py on a list of specs, at most WORKERS (<= 6) processes at once,
largest spec first. New in Part A'. LIST holds one "SPEC_PATH OUT_PATH" per line; an OUT_PATH that exists is skipped.
Each process's output goes to LOG_DIR/<out name>.log. Prints a summary.
Usage: python -B run_leaves_a2.py LIST LOG_DIR [--workers N]
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PY, TREE  # noqa: E402


def main():
    args = sys.argv[1:]
    workers = 6
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i + 1])
        del args[i:i + 2]
    if workers > 6:
        raise SystemExit("at most six workers")
    lst, logd = Path(args[0]), Path(args[1])
    logd.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(x.split()) for x in lst.read_text().splitlines() if x.strip()]
    todo = [(Path(s), Path(o)) for s, o in pairs if not Path(o).exists()]
    todo.sort(key=lambda p: -p[0].stat().st_size)
    env = dict(os.environ, PYTHONPATH=f"{TREE}:{HERE}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    running, rc = {}, {}
    t0 = time.time()
    while todo or running:
        while todo and len(running) < workers:
            s, o = todo.pop(0)
            o.parent.mkdir(parents=True, exist_ok=True)
            log = open(logd / f"{o.parent.name}_{o.stem}.log", "w")
            running[o] = subprocess.Popen([PY, "-B", str(HERE / "leaf_a2.py"), str(s), str(o)], cwd=str(TREE), env=env,
                                          stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
        for o, p in list(running.items()):
            if p.poll() is not None:
                rc[str(o)] = p.returncode
                del running[o]
        time.sleep(0.2)
    bad = {k: v for k, v in rc.items() if v != 0}
    print(f"{len(rc)} specs run, {len(bad)} nonzero exits {bad}; wall {time.time() - t0:.0f} s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
