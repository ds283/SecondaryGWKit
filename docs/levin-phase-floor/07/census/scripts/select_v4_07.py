"""levin-phase-floor prompt 07 section 6, V4: select the 24 stationary-point group calls, from 06m's census and the
rows alone (no result at g is read). New in 07. Read only. The rule (fixed in state/predictions.md before the census):

- **The population:** every group call of 06m's census (levin-phase-floor-06m-census-20261106T102153, checkpoint
  ea5bc3ab..., checked) at 1e-7 whose 06a scan entry (the unit's ``scan``, which 06m's census re-ran on the call's own
  theta_deriv) shows a sign change of theta' (``n_sign_changes`` > 0), and the same call (the same row, call index,
  label and span, checked) at 1e-8. These are the 720 rows' calls; less z00 and z12 (g-1e-7-1-2174 #1 and
  g-1e-8-1-2174 #1), which V4 scores already.
- Each call carries its band and k (state/rows.json: ``band``, ``k_key`` and ``k``), its rtol, and 06m's ``max_depth``
  and ``n_at_depth_max``.
- **The deepest four:** the population sorted by 06m's depth, deepest first, then by 06m's ``n_at_depth_max`` (most
  first), then by (rtol, shard, serial, index) ascending; the first four calls of distinct rows.
- **The other twenty:** four cells, band x rtol, (low, 1e-7), (low, 1e-8), (mid, 1e-7), (mid, 1e-8), five calls each.
  The eight k (all eight carry a stationary-point call; the k are ordered by value, k1 = 1e5 ... k8 = 3e8) are put two
  per k: k1, k3, k5, k7 in (low, 1e-7) and (mid, 1e-8); k2, k4, k6, k8 in (low, 1e-8) and (mid, 1e-7). Each cell's
  fifth slot takes the k with the most calls of the cell's population not yet slotted in the cell (ties: smaller k).
  Each cell's five slots, in ascending k, take the cell population's 06m depth quantiles 0.1, 0.5, 0.9, 0.3, 0.7 in
  that order (the quantile q of the cell's depths, sorted ascending, is the element at index floor(q (n - 1) + 1/2)).
  Each slot takes the call of its cell and k whose 06m depth is nearest its target (ties: (shard, serial, index)
  ascending), not among the deepest four and of a row not already chosen.

Writes OUT_JSON with the header, the population's counts and the 24 calls v00..v23 (deepest four first), each with
its unit, rtol, kind, index, label, band, k, 06m's depth, n_at_depth_max, converged, value and abserr (hex) and why it
was chosen. Refuses to overwrite.
Usage: python -B select_v4_07.py OUT_JSON
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06M_SHA256, RUN06M, WIDER, header, sha256  # noqa: E402

ROWS = WIDER / "state" / "rows.json"
ROWS_SHA256 = "aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417"
ALREADY = {("1e-7", "1-2174", 1), ("1e-8", "1-2174", 1)}  # z00, z12
QUANTILES = (0.1, 0.5, 0.9, 0.3, 0.7)
CELLS = (("low", "1e-7"), ("low", "1e-8"), ("mid", "1e-7"), ("mid", "1e-8"))


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists; not overwritten")
    ck = RUN06M / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: V4's 24 stationary-point group calls", [ck, ROWS])
    if sha256(ROWS) != ROWS_SHA256 or sha256(ck) != CHECKPOINT06M_SHA256:
        raise SystemExit("rows.json or 06m's checkpoint has another SHA-256")
    rows = {r["name"]: r for r in json.loads(ROWS.read_text())["rows"]}
    calls, sp = {}, []
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            for c in d["calls"]:
                calls[(d["rtol"], d["row"], c["index"])] = (r["unit"], c)
            if d["rtol"] == "1e-7":
                for s in d["scan"] or []:
                    if s is None:
                        raise SystemExit(f"{r['unit']}: a scan entry is None")
                    if s["n_sign_changes"] > 0:
                        sp.append((d["row"], s["index"]))
    pop = []
    for row, i in sp:
        u7, c7 = calls[("1e-7", row, i)]
        u8, c8 = calls[("1e-8", row, i)]
        if c7["notify_label"] != c8["notify_label"] or c7["x_span"] != c8["x_span"]:
            raise SystemExit(f"{row} #{i}: the 1e-7 and 1e-8 calls differ in label or span")
        for rt, u, c in (("1e-7", u7, c7), ("1e-8", u8, c8)):
            if (rt, row, i) in ALREADY:
                continue
            R = rows[row]
            pop.append({"unit": u, "rtol": rt, "row": row, "shard": R["shard"], "serial": R["serial"], "kind": "group",
                        "index": i, "label": c["notify_label"], "group": c["group"], "band": R["band"],
                        "k_key": R["k_key"], "k": R["k"], "depth_06m": c["max_depth"],
                        "n_at_depth_max_06m": c["n_at_depth_max"], "converged_06m": c["converged"],
                        "value_06m": c["value"]["hex"], "abserr_06m": c["abserr"]["hex"]})
    ks = sorted({p["k"] for p in pop}, key=float)
    kname = {k: f"k{j + 1}" for j, k in enumerate(ks)}

    def tie(p):
        return (p["rtol"], p["shard"], p["serial"], p["index"])

    chosen, rows_used = [], set()
    for p in sorted(pop, key=lambda p: (-p["depth_06m"], -p["n_at_depth_max_06m"], tie(p))):
        if p["row"] not in rows_used:
            chosen.append(dict(p, why="deepest four"))
            rows_used.add(p["row"])
        if len(chosen) == 4:
            break
    deepest = {(p["rtol"], p["row"], p["index"]) for p in chosen}
    plan = {CELLS[0]: [ks[0], ks[2], ks[4], ks[6]], CELLS[3]: [ks[0], ks[2], ks[4], ks[6]],
            CELLS[1]: [ks[1], ks[3], ks[5], ks[7]], CELLS[2]: [ks[1], ks[3], ks[5], ks[7]]}
    cells_doc = {}
    for cell in CELLS:
        band, rt = cell
        cp = [p for p in pop if p["band"] == band and p["rtol"] == rt]
        slotted = list(plan[cell])
        rest = [k for k in ks if k not in slotted]
        fifth = max(rest, key=lambda k: (sum(1 for p in cp if p["k"] == k), -float(k)))
        slots = sorted(slotted + [fifth], key=float)
        depths = sorted(p["depth_06m"] for p in cp)
        targets = [depths[int(math.floor(q * (len(depths) - 1) + 0.5))] for q in QUANTILES]
        cells_doc[f"{band}/{rt}"] = {"n": len(cp), "slots": [kname[k] for k in slots], "fifth": kname[fifth],
                                     "targets": targets}
        for k, q, tgt in zip(slots, QUANTILES, targets):
            cand = [p for p in cp if p["k"] == k and (p["rtol"], p["row"], p["index"]) not in deepest
                    and p["row"] not in rows_used]
            if not cand:
                raise SystemExit(f"no candidate for {cell} {kname[k]}")
            best = min(cand, key=lambda p: (abs(p["depth_06m"] - tgt), p["shard"], p["serial"], p["index"]))
            chosen.append(dict(best, why=f"cell {band}/{rt}, {kname[k]}, depth quantile {q} (target {tgt})"))
            rows_used.add(best["row"])
    for j, p in enumerate(chosen):
        p["sid"] = f"v{j:02d}"
        p["k_name"] = kname[p["k"]]
    counts = {
        "population": len(pop),
        "by_band_rtol": {f"{b}/{r}": sum(1 for p in pop if p["band"] == b and p["rtol"] == r) for b, r in CELLS},
        "by_k": {kname[k]: sum(1 for p in pop if p["k"] == k) for k in ks},
        "k_values": {kname[k]: k for k in ks},
        "bands_present": sorted({p["band"] for p in pop}),
    }
    doc = {"header": hdr, "rule": __doc__, "counts": counts, "cells": cells_doc, "calls": chosen}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps(counts))
    print(json.dumps(cells_doc))
    for p in chosen:
        print(f"{p['sid']} {p['unit']} #{p['index']} {p['rtol']} {p['band']} {p['k_name']} depth {p['depth_06m']} "
              f"at-cap {p['n_at_depth_max_06m']} conv {p['converged_06m']} | {p['why']}")
    print(f"strata: bands {sorted({p['band'] for p in chosen})}; k {sorted({p['k_name'] for p in chosen})}; rtols "
          f"{sorted({p['rtol'] for p in chosen})}; distinct rows {len({p['row'] for p in chosen})}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
