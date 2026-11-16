"""levin-phase-floor prompt 07b Part A, section 2.2: the spec of one call's piece references (leafref_07b.py's input).
New in 07b.

For each leaf named ``VARIANT:I`` (leaf I of out/replays/<VARIANT>/<SID>.json), its span and its break points: the
halves' (c = (b + a)/2), the quarters' (cL = (c + a)/2, cR = (b + c)/2) and, for a leaf named with a trailing ``:8``, the
eighths' (each quarter's (hi + lo)/2): the expressions of ``_levin_leaf.break_point`` and ``_check_leaf``, so that the
pieces are the solves' spans exactly. ``--all-leaves VARIANT`` adds every returned leaf's ends of that replay (for a
whole-call attribution: its spans are then every leaf). Every replay named must be of the same capture.
Writes OUT_SPEC. Usage: python -B leafspec_07b.py SID OUT_SPEC LEAF [LEAF ...] [--all-leaves VARIANT]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("out_spec")
    ap.add_argument("leaves", nargs="*")
    ap.add_argument("--all-leaves", default=None)
    a = ap.parse_args()
    out_p = Path(a.out_spec)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    reps, cap = {}, None
    points, spans, named = set(), [], []

    def rep(v):
        if v not in reps:
            reps[v] = json.loads((ROOT / "out" / "replays" / v / f"{a.sid}.json").read_text())
        return reps[v]

    for name in a.leaves:
        parts = name.split(":")
        v, i = parts[0], int(parts[1])
        eighths = len(parts) > 2 and parts[2] == "8"
        r = rep(v)
        c = r["header"]["capture"]
        key = (c["path"], c["offset"], c["sha256"], json.dumps(c["override"], sort_keys=True))
        if cap is None:
            cap = (key, c)
        elif cap[0] != key:
            raise SystemExit("the replays are not of one capture")
        L = r["leaves"][i]
        lo, hi = float.fromhex(L["start"]), float.fromhex(L["end"])
        cm = (hi + lo) / 2.0
        cL, cR = (cm + lo) / 2.0, (hi + cm) / 2.0
        pts = [lo, hi, cm, cL, cR]
        if eighths:
            for q0, q1 in ((lo, cL), (cL, cm), (cm, cR), (cR, hi)):
                pts.append((q1 + q0) / 2.0)
        points |= set(pts)
        spans.append((lo, hi))
        named.append({"name": name, "variant": v, "i": i, "lo": L["start"], "hi": L["end"], "depth": L["depth"],
                      "type": L["type"], "eighths": eighths})
    if a.all_leaves:
        r = rep(a.all_leaves)
        c = r["header"]["capture"]
        if cap is None:
            cap = (None, c)
        for L in r["leaves"]:
            lo, hi = float.fromhex(L["start"]), float.fromhex(L["end"])
            points |= {lo, hi}
            spans.append((lo, hi))
    spans = sorted(set(spans))
    spec = {"sid": a.sid, "capture": cap[1], "spans": [[s0.hex(), s1.hex()] for s0, s1 in spans],
            "points": [p.hex() for p in sorted(points)], "named": named, "all_leaves": a.all_leaves}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(spec, indent=1))
    print(f"{a.sid}: {len(named)} named leaves, {len(spans)} spans, {len(points)} points; wrote {out_p}")


if __name__ == "__main__":
    main()
