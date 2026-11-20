"""Compare B2's R and E ladders re-run at HEAD (cells/) with B2's registered run.

Reads docs/handover/b2-at-head/cells/*.json and B2's checkpoint,
var/runs/handover-13-residue-20261027T110832/checkpoint.jsonl, and prints one row per cell:
N + 9/8, Levin regions, max depth and integral time at each, the change in the cell's total
relative to it, and the number of group calls whose value moved by more than the sum of the two
declared errors (matched by label and z range).
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
CELLS = REPO / "docs/handover/b2-at-head/cells"
CHECKPOINT = REPO / "var/runs/handover-13-residue-20261027T110832/checkpoint.jsonl"
RUNGS = ["980", "10000", "100000", "1e+06", "1e+07", "1e+08"]
SHAPES = ["together", "T-first", "q-smooth"]

old = {}
for line in CHECKPOINT.read_text().splitlines():
    rec = json.loads(line)
    old[rec["unit"]] = rec["data"]


def moved(new_groups, old_groups):
    n = big = 0
    for g in new_groups:
        for o in old_groups:
            if g["label"] != o["label"]:
                continue
            if abs(g["z_min"] - o["z_min"]) > 1e-9 * g["z_min"]:
                continue
            if abs(g["z_max"] - o["z_max"]) > 1e-9 * g["z_max"]:
                continue
            gv, ga = g.get("raw_value", g["value"]), g.get("raw_abserr", g["abserr"])
            ov, oa = o.get("raw_value", o["value"]), o.get("raw_abserr", o["abserr"])
            if abs(gv - ov) > ga + oa:
                n += 1
                if abs(gv - ov) > 100 * (ga + oa):
                    big += 1
    return n, big


print(
    "| cell | N+9/8 B2 | N+9/8 HEAD | |dtotal/total| | regions B2 → HEAD | depth B2 → HEAD |"
    " integral time B2 → HEAD (s) | group calls moved > errors (> 100×) |"
)
print("|---|---|---|---|---|---|---|---|")
totals = {}
for obj in ("R", "E"):
    tb = th = 0.0
    for x in RUNGS:
        for shape in SHAPES:
            p = CELLS / f"{obj}-{shape}-{x}.json"
            if not p.exists():
                continue
            new = json.loads(p.read_text())
            o = old[f"{obj}|{shape}|{float(x):g}"]
            assert o["b2_cells_sha256"] == new["b2_cells_sha256"], p
            d = abs(new["total"] - o["total"]) / abs(o["total"])
            n, big = moved(new["groups"], o["groups"])
            tb += o["integral_time"]
            th += new["integral_time"]
            print(
                f"| {obj} {shape} {float(x):g} | {o['N_plus_9_8']:+.4e} | {new['N_plus_9_8']:+.4e} | {d:.1e} |"
                f" {o['Levin_regions']} → {new['Levin_regions']} | {o['max_depth']} → {new['max_depth']} |"
                f" {o['integral_time']:.1f} → {new['integral_time']:.1f} | {n} ({big}) |"
            )
    totals[obj] = (tb, th)
print()
for obj, (tb, th) in totals.items():
    print(f"{obj}: integral time B2 {tb:.1f} s, HEAD {th:.1f} s, ratio {th / tb:.1f}")
print()
print("growth of Levin regions, 980 to the top rung (HEAD; B2):")
for obj in ("R", "E"):
    for shape in SHAPES:
        xs = [x for x in RUNGS if (CELLS / f"{obj}-{shape}-{x}.json").exists()]
        a = json.loads((CELLS / f"{obj}-{shape}-{xs[0]}.json").read_text())[
            "Levin_regions"
        ]
        b = json.loads((CELLS / f"{obj}-{shape}-{xs[-1]}.json").read_text())[
            "Levin_regions"
        ]
        oa = old[f"{obj}|{shape}|{float(xs[0]):g}"]["Levin_regions"]
        ob = old[f"{obj}|{shape}|{float(xs[-1]):g}"]["Levin_regions"]
        print(
            f"  {obj} {shape} to {float(xs[-1]):g}: HEAD {a} → {b} ({b / a:.3g}×); B2 {oa} → {ob} ({ob / oa:.3g}×)"
        )
sys.exit(0)
