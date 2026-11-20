"""Compare R|together at 980 and 1e8 at HEAD as run (cells/) with the same cells run with T's
Levin phase derivative taken from the phase (cells-phase/, cell_phase.py): per cell N + 9/8, the
total, regions, depth and integral time; per group call the regions, depth and
|value change| / (sum of the two declared errors)."""

import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
for x in ("980", "1e+08"):
    base = json.loads((D / "cells" / f"R-together-{x}.json").read_text())
    phase = json.loads((D / "cells-phase" / f"R-together-{x}.json").read_text())
    print(f"R together {x}")
    for name, d in (("as run", base), ("phase", phase)):
        print(
            f"  {name:6s} N+9/8 {d['N_plus_9_8']:+.6e}  total {d['total']:.17e}  abserr {d['total_abserr']:.3e}"
            f"  regions {d['Levin_regions']}  depth {d['max_depth']} ({d['max_depth_group']})"
            f"  integral {d['integral_time']:.1f} s  load {d['loadavg_before'][0]:.1f}"
        )
    print(
        f"  |dtotal/total| {abs(phase['total'] - base['total']) / abs(base['total']):.2e}"
    )
    pg = {(g["label"], round(g["z_min"], 6)): g for g in phase["groups"]}
    for g in base["groups"]:
        q = pg[(g["label"], round(g["z_min"], 6))]
        r = abs(g["raw_value"] - q["raw_value"]) / (g["raw_abserr"] + q["raw_abserr"])
        print(
            f"    {g['label']:6s} z_min {g['z_min']:.4g}  regions {g['num_regions']} -> {q['num_regions']}"
            f"  depth {g['max_depth']} -> {q['max_depth']}  |dv|/(err+err) {r:.3g}"
        )
