"""06f review, check 2 (the orchestrator's own): replay Z2's call (g-1e-8-0-33995 call 5) from the
agent's capture, with nothing wrapped, at the tree on PYTHONPATH, and compare it with 06e's census:
value, abserr, num_regions, max_depth, converged (checkpoint), and every returned region's
start, end, depth, type, abserr, phase_err, phase_limited (06e's regions pickle, record checked by
its SHA-256). Also compares the regions' spans with the agent's per-leaf table.

Usage: PYTHONPATH=<tree g>:<06f scripts> python -B orch_z2_replay.py   (one process, no Ray)"""

import gzip
import hashlib
import json
import pickle
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
V = MAIN / "var/levin-phase-floor-06f"
RUN06E = MAIN / "var/runs/levin-phase-floor-06e-census-20261030T081318"
UNIT, INDEX = "g-1e-8-0-33995", 5


def main():
    import ray.cloudpickle as cp
    import AdaptiveLevin.levin_quadrature as LQ

    tree = LQ.__file__.split("/AdaptiveLevin/")[0]
    git = ["git", "--no-optional-locks", "-C", tree]
    print("# host", socket.gethostname(), "utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print("# tree", tree, subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
          "status lines", len(subprocess.run(git + ["status", "--short"], capture_output=True, text=True).stdout.splitlines()))
    print("# script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print("# levin_quadrature sha256", hashlib.sha256(Path(LQ.__file__).read_bytes()).hexdigest())
    ck = RUN06E / "checkpoint.jsonl"
    print("# 06e checkpoint sha256", hashlib.sha256(ck.read_bytes()).hexdigest())

    cap = [e for e in json.loads((V / "out/capture.json").read_text()) if e["unit"] == UNIT and e["index"] == INDEX][0]
    with open(Path(cap["dir"]) / cap["pickle"], "rb") as f:
        f.seek(cap["offset"])
        blob = f.read(cap["length"])
    sha = hashlib.sha256(blob).hexdigest()
    print("# capture", cap["dir"], cap["pickle"], "sha256", sha, "matches capture.json", sha == cap["sha256"])
    rec = cp.loads(blob)
    print("# capture keys", sorted(rec))

    t0 = time.time()
    rr = LQ.adaptive_levin_sincos(tuple(rec["x_span"]), rec["f"], theta=rec["theta"], **rec["kw"])
    wall = time.time() - t0
    print("# replay wall_s", round(wall, 1))

    d = c = None
    with open(ck) as f:
        for line in f:
            if f'"unit": "{UNIT}"' not in line[:300]:
                continue
            d = json.loads(line)["data"]
            break
    for x in (d.get("calls") or []) + (d.get("other_calls") or []):
        if x["index"] == INDEX:
            c = x
    got = {"value": float(rr["value"]).hex(), "abserr": float(rr["abserr"]).hex(), "num_regions": rr["num_regions"],
           "max_depth": rr["max_depth"], "converged": bool(rr["converged"])}
    want = {"value": c["value"]["hex"], "abserr": c["abserr"]["hex"], "num_regions": c["num_regions"],
            "max_depth": c["max_depth"], "converged": bool(c["converged"])}
    print("replay ", json.dumps(got))
    print("census ", json.dumps(want))
    print("scalars equal:", got == want)
    print("x_span equal:", [float(v).hex() for v in rec["x_span"]] == [h["hex"] for h in c["x_span"]],
          "label equal:", rec.get("notify_label", c["notify_label"]) == c["notify_label"])

    with open(RUN06E / d["pickle"], "rb") as f:
        f.seek(d["pickle_offset"])
        pblob = f.read(d["pickle_length"])
    print("# census regions record sha256 matches checkpoint:", hashlib.sha256(pblob).hexdigest() == d["pickle_record_sha256"])
    prec = pickle.loads(pblob)
    arr = [r for r in prec["regions"] + prec["other_regions"] if r["index"] == INDEX][0]
    regs = rr["regions"]
    n = len(regs)
    print("regions: replay", n, "census", len(arr["start"]))
    fields = {"start": lambda r: r.start, "end": lambda r: r.end, "depth": lambda r: r.depth, "type": lambda r: r.type,
              "abserr": lambda r: r.abserr, "phase_err": lambda r: r.phase_err, "phase_limited": lambda r: r.phase_limited,
              "total_err": lambda r: r.total_err}
    allsame = n == len(arr["start"])
    for name, get in fields.items():
        bad = 0
        for i, r in enumerate(regs):
            a, b = get(r), arr[name][i]
            if name == "type":  # r.type is "Levin" or "direct"; the census stores 0 / 1 (worker.py region_arrays)
                same = (0 if a == "Levin" else 1) == int(b)
            elif isinstance(a, bool) or name == "depth":
                same = int(a) == int(b)
            else:
                same = (a is None and b != b) or (a is not None and float(a).hex() == float(b).hex())
            bad += not same
        print(f"  {name:14s} mismatches {bad}")
        allsame &= bad == 0
    print("every region equal to the census, bit for bit:", allsame)

    leaves = json.load(gzip.open(V / "out/replay/g-1e-8-0-33995_5/leaves.json.gz", "rt"))
    spans_agent = sorted((L["a"]["hex"], L["b"]["hex"]) for L in leaves)
    spans_mine = sorted((float(r.start).hex(), float(r.end).hex()) for r in regs)
    print("agent's leaf table: rows", len(leaves), "spans equal to my replay's regions:", spans_agent == spans_mine)
    return 0 if (got == want and allsame and spans_agent == spans_mine) else 1


if __name__ == "__main__":
    sys.exit(main())
