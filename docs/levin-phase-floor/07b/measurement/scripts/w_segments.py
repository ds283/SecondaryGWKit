"""levin-phase-floor prompt 06j, Part W: per-segment phase spans of each W reference call, and the tool's chunks, for
the readings' backstop (state/readings_plan_*.md). New in 06j. For each call read (the W-5 rows' 1e-7 captures
f00-f04 and W-12's c00-c11): the tool's default edges (production_edges), each segment's |theta(X_r) - theta(X_l)|
(raw theta at the edges, as w_prep.py), and the Route B chunks the tool forms (ROUTE_B_CHUNK segments each, in
order), each chunk's segment count and phase span. Writes out/w_segments.json. Reads only.

Usage: PYTHONPATH=<tree>:<this directory> python -B w_segments.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

SIDS = ["f00", "f01", "f02", "f03", "f04"] + [f"c{i:02d}" for i in range(12)]


def main():
    import ray.cloudpickle as cp
    import tools.filon_reference as F

    pk = [RC.ROOT / "state" / "inputs" / f"{s}.pkl" for s in SIDS]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: W calls' segments and chunks", pk + [Path(F.__file__)])
    out = {"header": hdr, "ROUTE_B_CHUNK": F.ROUTE_B_CHUNK, "calls": {}}
    for s, p in zip(SIDS, pk):
        call = cp.loads(p.read_bytes())
        if F.route_of(call) != "B":
            raise SystemExit(f"{s}: not Route B")
        edges, _ = F.production_edges(call)
        th = [float(call["theta"]["theta"](x)) for x in edges]
        spans = [abs(th[i + 1] - th[i]) for i in range(len(edges) - 1)]
        n = len(spans)
        bounds = list(range(0, n, F.ROUTE_B_CHUNK)) + [n]
        chunks = [[bounds[i], bounds[i + 1], float(sum(spans[bounds[i]:bounds[i + 1]]))] for i in range(len(bounds) - 1)]
        out["calls"][s] = {"n_segments": n, "phase_span": float(sum(spans)), "segment_spans": spans, "chunks": chunks}
        big = sorted(chunks, key=lambda c: -c[2])[:3]
        print(f"{s}: {n} segments, {len(chunks)} chunks, span {sum(spans):.4g}; largest chunks {[(c[0], round(c[2] / max(1e-300, sum(spans)), 4)) for c in big]}")
    (RC.ROOT / "out" / "w_segments.json").write_text(json.dumps(out))
    print(f"wrote out/w_segments.json sha256 {RC.sha256(RC.ROOT / 'out' / 'w_segments.json')}")


if __name__ == "__main__":
    main()
