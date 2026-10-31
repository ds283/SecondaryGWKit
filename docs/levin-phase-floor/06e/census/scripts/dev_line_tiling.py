"""levin-phase-floor 06e, a development check of worker.py's _line_tiling (new in 06e): Z0's units had
no call past depth 12, so the tiling code was not exercised there. A synthetic deep line is built
from depth20-tiling's sample 1 span (g-1e-7-2-1444 call 4) by bisecting at (b + a) / 2.0 to the
left 20 times, in the levels' format _deep_line writes ("start"/"end" hex), and _line_tiling is run
on it with the tree on PYTHONPATH's chebyshev_matrices at order 24 (and 47). Expected: at g every
level tiles and is exact; at p some do not. Prints the per-level rows. Read-only; no Ray.
Usage: PYTHONPATH=<tree> python -B dev_line_tiling.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

import worker  # noqa: E402

LQ = importlib.import_module("AdaptiveLevin.levin_quadrature")
print("levin_quadrature", LQ.__file__)
a, b = float.fromhex("0x1.c1a02530aec6fp+0"), float.fromhex("0x1.8b0e134cd0b2cp+4")  # sample 0's span
levels = []
for d in range(21):
    levels.append({"start": a.hex(), "end": b.hex()})
    b = (b + a) / 2.0
rows = worker._line_tiling(LQ, {"levels": levels}, 24)
for d, r in enumerate(rows):
    print(d, r)
print("levels", len(rows), "tile at N on all:", all(r[0] for r in rows), "exact at N on all:", all(r[1] for r in rows),
      "tile at 2N-1 on all:", all(r[2] for r in rows), "exact at 2N-1 on all:", all(r[3] for r in rows))
