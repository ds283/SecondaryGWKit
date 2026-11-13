"""Why R30's block captures are not byte-identical to R29's (levin-phase-floor 07a, rehearsal agent, R30). New.
Loads each pair (R30's state/inputs/{blk}-h-*.pkl, R29's {blk}-g-*.pkl) with ray.cloudpickle at the tree h (Ray not
initialised) and lists the top-level keys whose values differ, comparing non-callables by repr (floats by hex) and
saying which keys hold callables (those are compared by block_score_r30.py's 1001-point samples).

    PYTHONPATH=<tree h>:<scripts/capture> python -B capture_compare.py
"""
import hashlib
import json
import sys
from pathlib import Path

import ray.cloudpickle as cp

Z = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal")
Z29 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal/r29")
PAIRS = [("J4", "0-10129", "0-157"), ("Y4", "0-10129", "0-157"), ("J4", "1-20022", "1-10283"), ("Y4", "1-20022", "1-10283")]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(v):
    if isinstance(v, float):
        return v.hex()
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()} if not any(callable(x) for x in v.values()) else "<dict holding callables>"
    if callable(v):
        return "<callable>"
    return repr(v)


print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
import ray
for blk, sh, sg in PAIRS:
    a, b = Z / "state" / "inputs" / f"{blk}-h-{sh}.pkl", Z29 / "state" / "inputs" / f"{blk}-g-{sg}.pkl"
    print(f"# input {a} sha256 {sha256(a)} bytes {a.stat().st_size}")
    print(f"# input {b} sha256 {sha256(b)} bytes {b.stat().st_size}")
    da, db = cp.loads(a.read_bytes()), cp.loads(b.read_bytes())
    keys = sorted(set(da) | set(db))
    diff = {k: (norm(da.get(k)), norm(db.get(k))) for k in keys if norm(da.get(k)) != norm(db.get(k))}
    calls = [k for k in keys if norm(da.get(k)) in ("<callable>", "<dict holding callables>") or (isinstance(da.get(k), (list, tuple)) and any(callable(x) for x in da.get(k)))]
    print(f"{blk} R30 {sh} vs R29 {sg}: keys {keys}; differing non-callable keys {json.dumps(diff)}; keys holding callables {calls}")
print(f"ray initialized: {ray.is_initialized()}")
