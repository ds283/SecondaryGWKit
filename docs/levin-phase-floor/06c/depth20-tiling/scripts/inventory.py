"""depth20-tiling step 0: the inventory of the 262 depth-20 calls, before any result is read.

For each entry of the census's out/deep_lines_06c.json (read only for its identity fields: unit,
index, notify_label, max_depth), the census checkpoint's record of the unit and the call: the
tolerance, the source block (from notify_label), the unit's wall_s, the call's elapsed, x_span
(hex), atol, num_regions, converged. Nothing about residuals, floors or tiling is read here.
Writes state/inventory.json.

Usage: PYTHONPATH=<tree>:<scripts> python -B inventory.py
"""

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import CHECKPOINT, DEEP_LINES, STATE, block_of, header, sha256  # noqa: E402


def main():
    for l in header(Path(__file__).resolve(), "depth20-tiling: inventory of the depth-20 calls", [DEEP_LINES, CHECKPOINT]):
        print(l)
    deep = json.loads(DEEP_LINES.read_text())
    ids = [(e["unit"], e["index"], e["notify_label"], e["max_depth"]) for e in deep]
    units = {u for u, *_ in ids}
    recs = {}
    with open(CHECKPOINT) as f:
        for line in f:
            u = line[10:60].split('"', 1)[0]
            if u in units:
                r = json.loads(line)
                recs[r["unit"]] = r["data"]
    assert set(recs) == units, units - set(recs)
    inv = []
    for u, idx, label, md in ids:
        d = recs[u]
        cands = [c for c in d["calls"] + d["other_calls"] if c["index"] == idx and c["notify_label"] == label]
        assert len(cands) == 1, (u, idx, len(cands))
        c = cands[0]
        rtol = u.split("-")[1] + "-" + u.split("-")[2]
        assert rtol == d["rtol"], (u, d["rtol"])
        inv.append({
            "unit": u, "index": idx, "notify_label": label, "block": block_of(label), "rtol": rtol,
            "kind": "group" if c in d["calls"] else "analytic",
            "unit_wall_s": d["wall_s"], "call_elapsed_s": c["elapsed"], "max_depth": md, "call_max_depth": c["max_depth"],
            "x_span": [c["x_span"][0]["hex"], c["x_span"][1]["hex"]], "rtol_call": c["rtol"], "atol": c["atol"].hex() if isinstance(c["atol"], float) else c["atol"],
            "num_regions": c["num_regions"], "converged": c["converged"], "value": c["value"]["hex"], "abserr": c["abserr"]["hex"],
            "evaluations": c["evaluations"],
        })
    out = STATE / "inventory.json"
    out.write_text(json.dumps(inv, indent=1))
    print(f"calls {len(inv)}; units {len(units)}")
    tab = collections.Counter((e["rtol"], e["block"]) for e in inv)
    print("calls by (rtol, block):")
    for rt in ("1e-7", "1e-8"):
        print("  " + rt + ": " + ", ".join(f"{b} {tab[(rt, b)]}" for b in ["G+q+r", "G+q-r", "G-q+r", "G-q-r", "J1", "J2", "J3", "J4", "Y1", "Y2", "Y3", "Y4"]))
    other = {k: v for k, v in tab.items() if k[1] not in ["G+q+r", "G+q-r", "G-q+r", "G-q-r", "J1", "J2", "J3", "J4", "Y1", "Y2", "Y3", "Y4"]}
    print(f"  other blocks: {other}")
    ws = sorted(e["unit_wall_s"] for e in inv)
    print(f"unit wall_s over the calls: min {ws[0]:.1f} median {ws[len(ws)//2]:.1f} max {ws[-1]:.1f}")
    print(f"wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
