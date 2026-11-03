"""Orchestrator check 2 (06j review): replay s17 at the tree with nothing wrapped and compare it with
06g's census checkpoint and region pickle, located from the checkpoint's own record (not 06j's selection).
PYTHONPATH=<worktree>:<06j scripts> python -B c2_replay_s17.py
"""

import hashlib
import json
import pickle
import sys

import numpy as np
import ray
import ray.cloudpickle as cp

import AdaptiveLevin.levin_quadrature as LQ

ROOT = "/Users/ds283/Documents/Code/SecondaryGWKit/var"
RUN = f"{ROOT}/runs/levin-phase-floor-06g-census-20261101T114315"
UNIT, INDEX, SID = "g-1e-8-1-35480", 2, "s17"

rec = None
with open(f"{RUN}/checkpoint.jsonl") as f:
    for line in f:
        if f'"unit": "{UNIT}"' in line:
            d = json.loads(line)
            if d["unit"] == UNIT:
                rec = d["data"]
call = [c for c in rec["calls"] if c["index"] == INDEX][0]

with open(f"{RUN}/{rec['pickle']}", "rb") as f:
    f.seek(rec["pickle_offset"])
    blob = f.read(rec["pickle_length"])
assert hashlib.sha256(blob).hexdigest() == rec["pickle_record_sha256"], "pickle record sha"
prec = pickle.loads(blob)
reg = [x for x in prec["regions"] if x["index"] == INDEX]
assert len(reg) == 1
reg = reg[0]

cap = cp.loads(open(f"{ROOT}/levin-phase-floor-06j/state/inputs/{SID}.pkl", "rb").read())
assert cap["unit"] == UNIT and cap["index"] == INDEX
res = LQ.adaptive_levin_sincos(cap["x_span"], cap["f"], cap["theta"], **cap["kw"])

out = {
    "value": [float(res["value"]).hex(), call["value"]["hex"]],
    "abserr": [float(res["abserr"]).hex(), call["abserr"]["hex"]],
    "num_regions": [int(res["num_regions"]), call["num_regions"]],
    "converged": [bool(res["converged"]), call["converged"]],
}
for k, (a, b) in out.items():
    print(f"{k}: replay {a} census {b} -> {'EQUAL' if a == b else 'DIFFER'}")
R = res["regions"]
for key, attr in (("start", "start"), ("end", "end"), ("value", "value"), ("depth", "depth")):
    mine = np.array([getattr(r, attr) for r in R])
    print(f"regions {key}: {len(mine)} vs {len(reg[key])} -> {'EQUAL' if np.array_equal(mine, reg[key]) else 'DIFFER'}")
cr_m = np.array([np.nan if r.convergence_ratio is None else r.convergence_ratio for r in R], dtype=float)
cr_c = np.asarray(reg["convergence_ratio"], dtype=float)
print(f"regions convergence_ratio: {'EQUAL' if np.array_equal(cr_m, cr_c, equal_nan=True) else 'DIFFER'}")
i = int(np.nanargmax(np.where(np.array([r.depth for r in R]) == 20, cr_m, np.nan)))
print(f"one returned leaf: #{i} [{R[i].start!r}, {R[i].end!r}] depth {R[i].depth} trusted {R[i].trusted} "
      f"r replay {cr_m[i].hex()} census {float(cr_c[i]).hex()}")
print(f"ray initialised: {ray.is_initialized()}")
