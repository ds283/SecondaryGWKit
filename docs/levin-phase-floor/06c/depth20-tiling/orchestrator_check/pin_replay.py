"""Orchestrator's own pinned-end-node replay (depth20-tiling re-check). Read-only.

Usage: PYTHONPATH=<tree e5764a6>:<dir with harness.py> python -B pin_replay.py PICKLE...
Replays each captured call as it is, then with chebyshev_matrices swapped in this process for a
version whose x[0] is b and x[-1] is a exactly (every other node and D untouched).
"""

import hashlib
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import ray.cloudpickle as cp

import AdaptiveLevin.levin_quadrature as LQ

tree = LQ.__file__.split("/AdaptiveLevin/")[0]
print("host", socket.gethostname(), "utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
print("tree", tree, subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
      "status lines", len(subprocess.run(["git", "-C", tree, "status", "--short"], capture_output=True, text=True).stdout.splitlines()))
print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())

ORIG = LQ.chebyshev_matrices


def pinned(x_span, N):
    x, D = ORIG(x_span, N)
    x = np.array(x, copy=True)
    x[0], x[-1] = x_span[1], x_span[0]
    return x, D


# the patch changes only the two ends
for span in [(0.1, 0.7), (1.0 / 3.0, 2.0 / 3.0), (-18.519, 9.5047), (5.5, 21.3)]:
    for N in (24, 47):
        x0, D0 = ORIG(span, N)
        x1, D1 = pinned(span, N)
        assert np.array_equal(x0[1:-1], x1[1:-1]) and np.array_equal(D0, D1) and x1[0] == span[1] and x1[-1] == span[0]
print("patch check: interior nodes and D unchanged, ends exact, on 4 spans x 2 orders")


def run(rec):
    t = time.perf_counter()
    r = LQ.adaptive_levin_sincos(tuple(rec["x_span"]), rec["f"], theta=rec["theta"], **rec["kw"])
    return r, time.perf_counter() - t


for p in sys.argv[1:]:
    blob = open(p, "rb").read()
    rec = cp.loads(blob)
    print(f"\n== {p} sha256 {hashlib.sha256(blob).hexdigest()[:16]}  span {rec['x_span'][0].hex()} .. {rec['x_span'][1].hex()}")
    r0, t0 = run(rec)
    print(f"  as is : value {r0['value'].hex()} abserr {r0['abserr'].hex()} regions {r0['num_regions']} depth {r0['max_depth']} conv {r0.get('converged')} {t0:.2f}s"
          f"  equals capture {(r0['value'].hex(), r0['abserr'].hex()) == (rec['value'], rec['abserr'])}")
    LQ.chebyshev_matrices = pinned
    try:
        r1, t1 = run(rec)
    finally:
        LQ.chebyshev_matrices = ORIG
    print(f"  pinned: value {r1['value'].hex()} abserr {r1['abserr'].hex()} regions {r1['num_regions']} depth {r1['max_depth']} conv {r1.get('converged')} {t1:.2f}s")
    print(f"  |dv|/abserr_census {abs(r1['value'] - r0['value']) / r0['abserr']:.3g}   |dv|/abserr_pinned {abs(r1['value'] - r0['value']) / r1['abserr']:.3g}")
