"""levin-phase-floor prompt 07c Part A, sections 2.2 and 2.3: write leafcc_07c.py's specs, one per call, and run them. New
in 07c.

Usage:
  python -B leafcc_specs_07c.py d1 D1_JSON OUT_DIR   the chase's Clenshaw-Curtis leaves: every leaf bisected strictly below a
      refused leaf whose route is the C2 gate's fallback (D1_JSON ``below``; the 4,009-type and 548-type of 07b), with its
      own value from its trust check; and every returned CC leaf below a refused leaf (``chase_cc_leaves``)
  python -B leafcc_specs_07c.py d2 OUT_DIR           the D2 sample (state/selection_d2.json)
  python -B leafcc_specs_07c.py run SPEC_DIR OUT_DIR [--workers 6]   run leafcc_07c.py on every spec, six at once
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"


def manifest():
    man = json.loads((ROOT / "state" / "calls_07c.json").read_text())["calls"]
    if (ROOT / "state" / "calls_07c_r5.json").exists():
        man.update(json.loads((ROOT / "state" / "calls_07c_r5.json").read_text())["calls"])
    return man


def cap(c):
    return {k: c[k] for k in ("path", "offset", "length", "sha256", "override")}


def write(out_dir, sid, leaves, man):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{sid.replace(':', '_')}.json"
    if p.exists():
        raise SystemExit(f"{p} exists")
    p.write_text(json.dumps({"sid": sid, "capture": cap(man[sid]), "leaves": leaves}))


def main():
    mode = sys.argv[1]
    man = manifest()
    if mode == "d1":
        d1 = json.loads(Path(sys.argv[2]).read_text())
        out_dir = Path(sys.argv[3])
        by = {}
        for x in d1["below"]:
            if x["route"] != "cc_gate":
                continue
            by.setdefault(x["sid"], {})[(x["start"], x["end"])] = {"id": f"b{x['seq']}", "start": x["start"], "end": x["end"],
                                                                   "forced": False, "value_own": x.get("own_value")}
        for sid, spans in d1["chase_cc_leaves"].items():
            for a, b in spans:
                by.setdefault(sid, {}).setdefault((a, b), {"id": f"f{a}", "start": a, "end": b, "forced": False, "value_own": None})
        n = 0
        for sid, d in sorted(by.items()):
            write(out_dir, sid, list(d.values()), man)
            n += len(d)
        print(f"{len(by)} specs, {n} leaves, in {out_dir}")
    elif mode == "d2":
        sel = json.loads((ROOT / "state" / "selection_d2.json").read_text())["selection"]
        out_dir = Path(sys.argv[2])
        by = {}
        for st, leaves in sel.items():
            for l in leaves:
                by.setdefault(l["sid"], []).append({"id": f"{st}|{l['start']}", "start": l["start"], "end": l["end"],
                                                    "forced": bool(l["window"]), "value_own": l["value_own"]})
        for sid, ls in sorted(by.items()):
            write(out_dir, sid, ls, man)
        print(f"{len(by)} specs, {sum(len(v) for v in by.values())} leaves, in {out_dir}")
    elif mode == "run":
        spec_dir, out_dir = Path(sys.argv[2]), Path(sys.argv[3])
        workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
        if workers > 6:
            raise SystemExit("at most six workers")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=f"{TREE}:{HERE}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
        pending = sorted(spec_dir.glob("*.json"))
        running, rc = {}, {}
        t0 = time.time()
        while pending or running:
            while pending and len(running) < workers:
                s = pending.pop(0)
                log = open(out_dir / "logs" / f"{s.stem}.log", "w")
                running[s.stem] = subprocess.Popen([PY, "-B", str(HERE / "leafcc_07c.py"), str(s), str(out_dir / s.name)],
                                                   cwd=TREE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                log.close()
            for k, p in list(running.items()):
                if p.poll() is not None:
                    rc[k] = p.returncode
                    del running[k]
            time.sleep(0.2)
        bad = {k: v for k, v in rc.items() if v != 0}
        print(f"{len(rc)} specs run in {time.time() - t0:.0f} s; nonzero exits: {bad}")
        return 1 if bad else 0
    else:
        raise SystemExit("d1, d2 or run")


if __name__ == "__main__":
    sys.exit(main())
