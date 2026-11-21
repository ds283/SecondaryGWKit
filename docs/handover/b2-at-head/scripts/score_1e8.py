"""Score the commit ladder's R|together|1e8 group calls against the Filon references.

Reads ``reference-1e8/readings-without-segments.jsonl`` (``ref_1e8.py``) and ``ladder/<sha>/R-*.json``
(``run_commit_ladder.sh``), and prints, as Markdown:

1. each reference: its four readings, their scatter, and the value (FCC at resolution 2);
2. per tree and call: regions, depth, converged, declared error, true error against the reference,
   and a verdict. A call is **honest** when ``|value - ref| + scatter <= abserr`` (it is within its
   declared error whatever the reference's own error), **dishonest** when
   ``|value - ref| - scatter > abserr``, and **undecided** otherwise. Values are the Levin call's
   own (``raw_value``, ``raw_abserr``); the reference is of that call;
3. per tree: the four calls' summed true error, in the cell's own units, relative to the cell total
   and as a change in N + 9/8, against the cell's declared relative error;
4. two determinism checks: the B2 tree's cell against B2's checkpoint, and 07c's against
   ``cells/R-together-1e+08.json`` (HEAD).

Usage: ``python score_1e8.py`` (paths are this record's own).
"""

import json
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[3]
LABELS = ("G+q+r", "G+q-r", "G-q+r", "G-q-r")
TREES = [
    ("d88faec8", "B2 (handover 13)"),
    ("4ddc3c2c", "06b"),
    ("ad9a5f48", "06c"),
    ("3d70db93", "06d"),
    ("f4d8e173", "06e"),
    ("7172a8f5", "06h"),
    ("3ae6f3a4", "06g"),
    ("48cf87c0", "06m"),
    ("ad732887", "07"),
    ("adffa21b", "07 (rule A)"),
    ("1ec734d7", "07a"),
    ("5e19c4b6", "07b"),
    ("67b5af74", "07c"),
]
CHECKPOINT = REPO / "var/runs/handover-13-residue-20261027T110832/checkpoint.jsonl"


def load_refs():
    refs = {}
    for line in (
        (ROOT / "reference-1e8/readings-without-segments.jsonl")
        .read_text()
        .splitlines()
    ):
        r = json.loads(line)
        refs.setdefault(r["label"], []).append(r)
    out = {}
    for label, rs in refs.items():
        vals = [mpf(r["reading"]["value_30"]) for r in rs]
        chosen = next(r for r in rs if r["impl"] == "fcc" and r["res"] == 2)
        out[label] = {
            "readings": rs,
            "value": mpf(chosen["reading"]["value_30"]),
            "scatter": max(vals) - min(vals),
            "z": (rs[0]["z_min"], rs[0]["z_max"]),
        }
    return out


def group(cell, label, z):
    hits = [
        g
        for g in cell["groups"]
        if g["label"] == label
        and abs(g["z_min"] - z[0]) <= 1e-9 * z[0]
        and abs(g["z_max"] - z[1]) <= 1e-9 * z[1]
    ]
    assert len(hits) == 1, (label, len(hits))
    return hits[0]


def g3(x):
    return f"{float(x):.3g}"


def main():
    refs = load_refs()
    print("## References (`reference-1e8/readings-without-segments.jsonl`)\n")
    print(
        "| call | z range | QAWO 1 | FCC 1 | QAWO 2 | FCC 2 (the value) | scatter | scatter / value |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for label in LABELS:
        R = refs[label]
        by = {(r["impl"], r["res"]): r["reading"]["value_30"] for r in R["readings"]}
        cols = [
            mp.nstr(mpf(by[k]), 16)
            for k in (("qawo", 1), ("fcc", 1), ("qawo", 2), ("fcc", 2))
        ]
        print(
            f"| `{label}` | ({R['z'][0]:.6g}, {R['z'][1]:.6g}) | "
            + " | ".join(cols)
            + f" | {g3(R['scatter'])} | {g3(R['scatter'] / abs(R['value']))} |"
        )

    print("\n## Each tree's calls against the references\n")
    print(
        "| tree | call | regions | depth | converged | declared / value | true error / value "
        "| true / declared | scatter / declared | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    cells = {}
    for sha, name in TREES:
        cell = json.loads((ROOT / f"ladder/{sha}/R-together-1e+08.json").read_text())
        cells[sha] = cell
        for label in LABELS:
            R = refs[label]
            g = group(cell, label, R["z"])
            v, a = mpf(g["raw_value"]), mpf(g["raw_abserr"])
            err = abs(v - R["value"])
            if err + R["scatter"] <= a:
                verdict = "honest"
            elif err - R["scatter"] > a:
                verdict = "**dishonest**"
            else:
                verdict = "undecided"
            print(
                f"| `{sha}` {name} | `{label}` | {g['num_regions']} | {g['max_depth']} | "
                f"{g['converged']} | {g3(a / abs(R['value']))} | {g3(err / abs(R['value']))} | "
                f"{g3(err / a)} | {g3(R['scatter'] / a)} | {verdict} |"
            )

    print("\n## Each tree's cell\n")
    print(
        "| tree | Levin regions | max depth | integral time | N + 9/8 | four calls' summed true error / total "
        "| reference scatter / total | declared (cell) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for sha, name in TREES:
        cell = cells[sha]
        total = mpf(cell["total"])
        s = sc = mpf(0)
        for label in LABELS:
            R = refs[label]
            g = group(cell, label, R["z"])
            scale = mpf(g["value"]) / mpf(g["raw_value"])
            s += (mpf(g["raw_value"]) - R["value"]) * scale
            sc += R["scatter"] * abs(scale)
        print(
            f"| `{sha}` {name} | {cell['Levin_regions']} | {cell['max_depth']} | "
            f"{cell['integral_time']:.1f} s | {cell['N_plus_9_8']:.12e} | {g3(s / total)} | "
            f"{g3(sc / abs(total))} | {g3(cell['declared'])} |"
        )

    print("\n## Determinism\n")
    old = None
    for line in CHECKPOINT.read_text().splitlines():
        rec = json.loads(line)
        if rec["unit"] == "R|together|1e+08":
            old = rec["data"]
    head = json.loads((ROOT / "cells/R-together-1e+08.json").read_text())
    for what, a, b in (
        ("`d88faec8` against B2's checkpoint", cells["d88faec8"], old),
        ("`67b5af74` against `cells/` (HEAD)", cells["67b5af74"], head),
    ):
        same_total = a["total"] == b["total"]
        same_groups = all(
            ga["raw_value"] == gb.get("raw_value", gb["value"])
            and ga["num_regions"] == gb["num_regions"]
            for ga, gb in zip(a["groups"], b["groups"])
        ) and len(a["groups"]) == len(b["groups"])
        print(
            f"- {what}: total identical {same_total}; every group's value and regions identical "
            f"{same_groups}; regions {a['Levin_regions']} and {b['Levin_regions']}"
        )


if __name__ == "__main__":
    main()
