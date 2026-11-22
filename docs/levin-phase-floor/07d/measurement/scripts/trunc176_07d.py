"""levin-phase-floor prompt 07d Part A, section 2.3: 07c's 176 truncation-dominated chase leaves, re-measured by this Part's
leaf method (leafcc_07d.py v3, reference v3) for the cause of understatement. New in 07d.

The leaves are 07c Part A's D1 leaves (var/levin-phase-floor-07c/measurement/out/leafcc_d1/, read only) that 07c's
d2_regimes_07c.py classes truncation-dominated: true error above 3 s_CC, s_CC = sigma sqrt(sum w^2) (176 of 22,876; all
resolved). Each is measured on 07c Part A's capture of its call (its state/leafcc_d1/<sid>.json, the blob's SHA-256 checked),
at the capture's rtol (07c's override where it had one). The leaves are 07b-code partitions' leaves; the Clenshaw-Curtis
solve is the same at 07c's code, which ``reproduces_cc`` checks against 07c's recorded value in hex.
Writes out/trunc176/specs/, out/trunc176/out/ and out/trunc176/index.json. Usage: PYTHONPATH=<tree>:<scripts> python -B trunc176_07d.py
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import MEAS07C, header, sha256  # noqa: E402

TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
fx = float.fromhex


def main():
    import AdaptiveLevin.levin_quadrature as LQ
    from replay_07d import load_capture

    od = ROOT / "out" / "trunc176"
    (od / "specs").mkdir(parents=True, exist_ok=False)
    (od / "out" / "logs").mkdir(parents=True, exist_ok=False)
    idx = {}
    for p in sorted((MEAS07C / "out" / "leafcc_d1").glob("*.json")):
        doc = json.loads(p.read_text())
        N = doc["N"]
        wb2 = float(np.sum(LQ._cc_weights_base(2 * N - 1) ** 2))
        leaves = []
        for l in doc["leaves"]:
            if "error" in l:
                continue
            w = abs(fx(l["end"]) - fx(l["start"]))
            sCC = fx(l["noise_sigma"]) * math.sqrt(wb2) * w / 2
            if fx(l["true_err"]) > 3 * sCC:
                a, b = sorted((fx(l["start"]), fx(l["end"])))
                leaves.append({"id": f"{p.stem}#{l['id']}", "start": a.hex(), "end": b.hex(), "type": "direct",
                               "value_own": l["value"] if a == fx(l["start"]) else None,
                               "d1_true_err": l["true_err"], "d1_ref_err": l["ref_err"], "d1_pair": l["candidates"]["pair"]})
        if not leaves:
            continue
        sp = json.loads((MEAS07C / "state" / "leafcc_d1" / p.name).read_text())
        cap = sp["capture"]
        rec = load_capture(cap)
        kw = dict(rec["kw"])
        if cap.get("override"):
            kw["rtol"], kw["atol"] = cap["override"]["rtol"], cap["override"]["atol"]
        spec = {"sid": f"07c-d1:{sp['sid']}", "capture": {k: cap[k] for k in ("path", "offset", "length", "sha256")},
                "rtol": kw["rtol"], "atol": kw["atol"], "call_value": rec["value"],
                "x_span": [float(rec["x_span"][0]).hex(), float(rec["x_span"][1]).hex()], "leaves": leaves}
        (od / "specs" / p.name).write_text(json.dumps(spec))
        idx[p.name] = len(leaves)
    env = dict(os.environ, PYTHONPATH=f"{TREE}:{HERE}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    pending = sorted((od / "specs").glob("*.json"))
    running, rc = {}, {}
    t0 = time.time()
    while pending or running:
        while pending and len(running) < 6:
            s = pending.pop(0)
            log = open(od / "out" / "logs" / f"{s.stem}.log", "w")
            running[s.stem] = subprocess.Popen([PY, "-B", str(HERE / "leafcc_07d.py"), str(s), str(od / "out" / s.name)],
                                               cwd=TREE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
        for k, p in list(running.items()):
            if p.poll() is not None:
                rc[k] = p.returncode
                del running[k]
        time.sleep(0.2)
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: 07c's 176 truncation-dominated chase leaves", [])
    (od / "index.json").write_text(json.dumps({"header": hdr, "n_leaves": sum(idx.values()), "by_call": idx, "exits": rc,
                                               "wall_s": time.time() - t0}, indent=1))
    print(f"{sum(idx.values())} leaves in {len(idx)} calls; nonzero exits {[k for k, v in rc.items() if v]}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
