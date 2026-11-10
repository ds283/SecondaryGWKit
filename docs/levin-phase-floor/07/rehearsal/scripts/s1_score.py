"""S1 -- V1's twelve (prompt 07 section 5.5), scored from R28's group records against 06g's references.

levin-phase-floor 07, rehearsal agent. New. Reads:
- 06g's references, docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json, key V1 (w00-w11:
  each row's (k, q, r, z_response), the four readings and ref_30, FCC at resolution 2, with 30 digits);
- R28's store (sqlite mode=ro): each w row's group record WKB_Levin.subintervals[1].groups[j] labelled
  G+q+r, its stored value and abserr ((1 + z_response) times the call's);
- the replay at g (state/capture/replay, the capture harness's emit files), whose call reproduces that record
  bit for bit (out/replay_calls.checks.json): the call's own value and abserr (unscaled, as 06g's reference
  is) and its located points (guard_07).

The rule (prompt 07 section 5.5): each locates its point at z ~ 10211, and |v28 - ref| <= abserr28. "Locates
its point at z ~ 10211" is read as: exactly one point, a root, whose z = exp(x) - 1 lies within one scan
spacing of z = 10211.3179 (P1's report section 4, row 1's z_s, to its printed digits), the spacing taken in z
at the point, (x_b - x_a)/4096 * (1 + z) (the location's 4097-point grid in x = log(1 + z)). Ratios in mpmath
at 40 digits from ref_30 and the doubles.

    ./venv/bin/python -I -B s1_score.py
"""

import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

import mpmath as mp

R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal")
sys.path.insert(0, str(R / "scripts" / "capture"))
import replay07 as RP  # noqa: E402

SCORES = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035/docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json")
Z_STAR = 10211.3179


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    mp.mp.dps = 40
    print(f"# script {Path(__file__).resolve()} sha256 {sha256(__file__)}")
    print(f"# input {SCORES} sha256 {sha256(SCORES)}")
    for f in sorted(RP.STORES["g"].parent.glob("levin-phase-floor-07-r28*.sqlite")):
        print(f"# input {f} sha256 {sha256(f)}")
    checks = R / "out" / "replay_calls.checks.json"
    print(f"# input {checks} sha256 {sha256(checks)}")
    v1 = json.loads(SCORES.read_text())["V1"]
    store = RP.read_store(RP.STORES["g"])
    units = RP.load_units(R / "state" / "capture" / "replay")
    chk = {(tuple(c["key"]), c["path"]): c for c in json.loads(checks.read_text())["checks"] if "path" in c}
    out = []
    holds = True
    for w in v1:
        key = (w["k"], w["q"], w["r"], w["z_response"])
        rec = store[key]
        sub = rec["metadata"]["WKB_Levin"]["subintervals"][1]
        js = [j for j, g in enumerate(sub["groups"]) if g["label"] == "G+q+r"]
        assert len(js) == 1, (w["wid"], js)
        path = f"WKB_Levin.subintervals[1].groups[{js[0]}]"
        g = sub["groups"][js[0]]
        sv, se = float(g["value"]).hex(), float(g["abserr"]).hex()
        c_ok = chk[(key, path)]["reproduces"]
        u, _, _ = units[f"g-1e-7-{rec['shard']}-{rec['serial']}"]
        hits = [c for c in u["calls"] if c["notify_label"].endswith("group G+q+r") and RP.stored_form(c, key[3]) == (sv, se)]
        assert len(hits) == 1, (w["wid"], len(hits))
        c = hits[0]
        gd = c["guard_07"]
        pts = gd["points"]
        xa, xb = float.fromhex(gd["x_span"][0]["hex"]), float.fromhex(gd["x_span"][1]["hex"])
        h = abs(xb - xa) / 4096
        v, e = float.fromhex(c["value"]["hex"]), float.fromhex(c["abserr"]["hex"])
        ref = mp.mpf(w["ref_30"])
        ratio = abs(mp.mpf(v) - ref) / mp.mpf(e)
        loc = None
        loc_ok = False
        if len(pts) == 1 and pts[0]["kind"] == "root":
            x = float.fromhex(pts[0]["x"]["hex"])
            z = math.exp(x) - 1.0
            dz = h * (1.0 + z)
            loc = {"x": pts[0]["x"]["hex"], "z": repr(z), "z_spacing": repr(dz), "abs_z_minus_10211.3179": repr(abs(z - Z_STAR)),
                   "theta_deriv": pts[0]["theta_deriv"]["repr"], "window_pieces": pts[0]["window_pieces"]}
            loc_ok = abs(z - Z_STAR) <= dz
        honest = bool(ratio <= 1)
        holds &= loc_ok and honest
        row = {"wid": w["wid"], "key": list(key), "record": path, "replay_reproduces_record": c_ok,
               "points": len(pts), "location": loc, "location_ok": loc_ok,
               "v28": c["value"]["hex"], "abserr28": c["abserr"]["hex"], "converged28": c["converged"],
               "regions28": c["num_regions"], "max_depth28": c["max_depth"], "store_value28": sv, "store_abserr28": se,
               "ref_30": w["ref_30"], "ref_hex": w["ref_hex"], "readings_spread_over_tol": w["spread_over_tol"],
               "ratio_v28": mp.nstr(ratio, 6), "honest": honest,
               "R24_ratio_06g": w["r24"], "R23_ratio_06g": w["r23"]}
        out.append(row)
        print(f"{w['wid']} ({key[0]:.6g}, {key[1]:.6g}, {key[2]:.6g}, z {key[3]!r}) {path}: replay reproduces {c_ok}; "
              f"points {len(pts)}" + (f" root at z {loc['z']} (|z - 10211.3179| {float(loc['abs_z_minus_10211.3179']):.3g}, "
                                      f"one spacing {float(loc['z_spacing']):.3g})" if loc else "") +
              f"; v28 {row['v28']} abserr28 {row['abserr28']} ref {w['ref_hex']} ({w['ref_30']}); "
              f"|v28 - ref|/abserr28 {row['ratio_v28']} ({'honest' if honest else 'NOT honest'}); converged {c['converged']}, "
              f"regions {c['num_regions']}, max depth {c['max_depth']}; 06g: R24 {w['r24']:.4g}, R23 {w['r23']:.4g}")
    worst = max(out, key=lambda r: float(r["ratio_v28"]))
    print(f"\nS1 {'HOLDS' if holds else 'FAILS'}: {sum(r['location_ok'] for r in out)} of 12 locate their point at z ~ 10211; "
          f"{sum(r['honest'] for r in out)} of 12 honest; largest |v28 - ref|/abserr28 {worst['ratio_v28']} ({worst['wid']}); "
          f"converged {sum(r['converged28'] for r in out)} of 12")
    dst = R / "out" / "s1_scores.json"
    dst.write_text(json.dumps({"holds": holds, "rows": out}, indent=1))
    print(f"wrote {dst} sha256 {sha256(dst)}")


if __name__ == "__main__":
    main()
