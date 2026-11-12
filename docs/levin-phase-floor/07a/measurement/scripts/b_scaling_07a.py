"""levin-phase-floor prompt 07a Part A: how variant (b)'s cost grows with depth_max, to project its cost at the calls'
own depth_max (20) before running it on section 2.1's calls. New in 07a; a cost probe, not a section 2.4 table.

One call, one variant, one depth_max per process: the capture loaded as replay_07a.py loads it, the variant installed
by 07's variant_loader, ``kw["depth_max"] = D`` (the only change), and the call's value, abserr, converged,
num_regions, max depth, num_window_pieces and wall recorded. Usage: PYTHONPATH=<tree> python -B b_scaling_07a.py
SID VARIANT D OUT_JSON
"""

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(1, str(HERE))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sid, variant, D, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    if out.exists():
        raise SystemExit(f"{out} exists")
    import variant_loader as VL

    vj = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"][variant]
    LQ = VL.install(vj["path"], vj["sha256"], f"07a_{variant}")
    import ray.cloudpickle as cp

    c = json.loads((ROOT / "state" / "calls_07a.json").read_text())["calls"][sid]
    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture sha256")
    rec = cp.loads(blob)
    kw = dict(rec["kw"])
    if c["override"]:
        kw["rtol"], kw["atol"] = c["override"]["rtol"], c["override"]["atol"]
    kw["depth_max"] = D
    t0 = time.perf_counter()
    r = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], rec["theta"], **kw)
    wall = time.perf_counter() - t0
    regs = r["regions"]
    doc = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "variant": variant,
           "variant_sha256": vj["sha256"], "sid": sid, "depth_max": D, "value": float(r["value"]).hex(),
           "abserr": float(r["abserr"]).hex(), "converged": bool(r["converged"]), "num_regions": int(r["num_regions"]),
           "max_depth": int(max(x.depth for x in regs)), "num_window_pieces": int(r["num_window_pieces"]),
           "n_window_at_depth_max": sum(1 for x in regs if x.window and x.depth >= D), "wall_s": wall}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc))


if __name__ == "__main__":
    main()
