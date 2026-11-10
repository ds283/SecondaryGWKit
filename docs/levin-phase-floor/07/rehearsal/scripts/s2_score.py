"""S2 -- P1's two rows (prompt 07 section 5.6), scored.

levin-phase-floor 07, rehearsal agent. New. Reads out/p1_references/calls.json (each call's pickle, R27's and
R28's call value and abserr, R28's guard_07 record from the replay) and the tool's outputs
out/p1_references/p1_row{1,2}.json (four readings, scatter, tolerance, counts, the reference = FCC at
resolution 2, value_30).

The rules (prompt 07 section 5.6):
- location: each locates its point within one scan spacing. P1's points (P1 report section 4) are
  x_s = 9.2313499 (z_s = 10211.3179) for row 1 and x_s = 5.6204320 (z_s = 275.00859) for row 2, x = log(1 + z);
  the scan spacing is the location's, (x_b - x_a)/4096 on its 4097-point grid in x. The point must be one
  root with |x - x_s| <= that spacing (x_s to P1's printed digits);
- honesty: |v28 - ref| <= abserr28, where the four readings agree (the tool's ``counts``: scatter at most the
  tolerance, min(abserr_R27, abserr_R28)/100). R27's distance is reported beside it.
Arithmetic in mpmath at 40 digits from value_30 and the doubles.

    ./venv/bin/python -I -B s2_score.py
"""

import hashlib
import json
import math
from pathlib import Path

import mpmath as mp

R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal")
P1 = {1: (9.2313499, 10211.3179), 2: (5.6204320, 275.00859)}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    mp.mp.dps = 40
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    cpath = R / "out" / "p1_references" / "calls.json"
    print(f"# input {cpath} sha256 {sha256(cpath)}")
    calls = json.loads(cpath.read_text())
    out = []
    holds = True
    for c in calls:
        rp = Path(c["out"])
        print(f"# input {rp} sha256 {sha256(rp)}")
        print(f"# input {c['pickle']} sha256 {sha256(c['pickle'])} (recorded {c['pickle_sha256']})")
        ref_doc = json.loads(rp.read_text())
        g = c["R28_guard_07"]
        xa, xb = float.fromhex(g["x_span"][0]["hex"]), float.fromhex(g["x_span"][1]["hex"])
        h = abs(xb - xa) / 4096
        pts = g["points"]
        xs, zs = P1[c["p1"]]
        loc_ok = len(pts) == 1 and pts[0]["kind"] == "root" and abs(float.fromhex(pts[0]["x"]["hex"]) - xs) <= h
        x = float.fromhex(pts[0]["x"]["hex"]) if pts else None
        ref = mp.mpf(ref_doc["value_30"])
        readings = {f"{r['impl']}_r{r['res']}": {"value_hex": r["value_hex"], "value_30": r["value_30"], "wall_s": r["wall_s"],
                                                "all_monotone": r.get("tmap", {}).get("all_monotone"),
                                                "direct_segments": r.get("tmap", {}).get("direct_segments")}
                    for r in ref_doc["readings"]}
        row = {"p1": c["p1"], "key": c["key"], "points": len(pts),
               "x": None if x is None else pts[0]["x"]["hex"], "x_repr": repr(x),
               "z": None if x is None else repr(math.exp(x) - 1.0), "P1_x_s": xs, "P1_z_s": zs,
               "abs_x_minus_x_s": None if x is None else repr(abs(x - xs)), "scan_spacing_x": repr(h),
               "location_ok": loc_ok, "readings": readings, "scatter": ref_doc["scatter"], "tolerance": ref_doc["tolerance"],
               "counts": ref_doc["counts"], "ref_hex": ref_doc["value_hex"], "ref_30": ref_doc["value_30"]}
        for tag in ("R27", "R28"):
            v, e = mp.mpf(float.fromhex(c[f"{tag}_value"])), mp.mpf(float.fromhex(c[f"{tag}_abserr"]))
            d = abs(v - ref)
            row[tag] = {"value": c[f"{tag}_value"], "abserr": c[f"{tag}_abserr"], "abs_v_minus_ref": mp.nstr(d, 6),
                        "ratio": mp.nstr(d / e, 6), "honest": bool(d <= e)}
        honest_ok = row["counts"] and row["R28"]["honest"]
        row["honesty_ok"] = honest_ok
        holds &= loc_ok and honest_ok
        out.append(row)
        print(f"\nrow {c['p1']} {c['key']}: located {len(pts)} point(s)" + (f", {pts[0]['kind']} at x {x!r} (z {math.exp(x) - 1.0!r}); "
              f"|x - x_s| {abs(x - xs):.3g} against one spacing {h:.4g}: {'within' if loc_ok else 'NOT within'}" if pts else ""))
        for k, r in readings.items():
            print(f"  {k}: {r['value_hex']} ({r['value_30']}), {r['wall_s']:.0f} s, all_monotone {r['all_monotone']}, direct_segments {r['direct_segments']}")
        print(f"  scatter {row['scatter']:.3g} against tolerance {row['tolerance']:.4g}: counts {row['counts']}; reference (FCC r2) {row['ref_hex']} ({row['ref_30']})")
        for tag in ("R27", "R28"):
            x_ = row[tag]
            print(f"  {tag}: value {x_['value']} abserr {x_['abserr']}: |v - ref| {x_['abs_v_minus_ref']} = {x_['ratio']} x abserr ({'honest' if x_['honest'] else 'NOT honest'})")
    print(f"\nS2 {'HOLDS' if holds else 'FAILS'}: location {sum(r['location_ok'] for r in out)} of 2; honesty {sum(r['honesty_ok'] for r in out)} of 2")
    dst = R / "out" / "s2_scores.json"
    dst.write_text(json.dumps({"holds": holds, "rows": out}, indent=1))
    print(f"wrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
