"""levin-phase-floor prompt 06j, Part W: the readings' cost, projected from each call's phase span (the brief and
prompt section 1: "Cost is projected from each call's phase span ... not from its segment count"). New in 06j.
Reads only, except out/cost_projection.json.

1. **Calibration** on 06g's Route B readings (var/levin-phase-floor-06g/references/out/readings/,
   rereads/, readings_fcc_b5626ec_laptop/; read only), every reading that finished with a wall time: its phase span
   is the sum over its segments of ``t_span`` (the segment's phase span as the tool fitted it), its wall the
   reading's ``wall_s``. Per (impl, res) the rate wall/span in microseconds per radian is tabulated for readings
   whose span is at least 1e7 rad (where the per-radian work dominates), with its range; and the wall per segment
   for every reading (the span-independent part, which dominates small spans).
2. **Projection** for each W reading: wall = max(the per-segment part, the per-radian part), each from the
   calibration's **largest** observed rate (a conservative projection), with the call's phase span from
   out/w_prep_v2 (sum over production_edges segments of |theta(X_r) - theta(X_l)|; no segment shows a sign change
   of theta_deriv at 17 points) and its segment count. FCC's per-radian rate is reported too (06g found FCC's cost
   does not grow with the span; if the calibration shows otherwise it is used).

Usage: python -B cost_06j.py
"""

import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

REF = RC.MAIN / "var" / "levin-phase-floor-06g" / "references" / "out"


def main():
    files = sorted(glob.glob(str(REF / "readings" / "*.json")) + glob.glob(str(REF / "rereads" / "*.json"))
                   + glob.glob(str(REF / "readings_fcc_b5626ec_laptop" / "*.json")))
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: Part W's reading costs, projected from phase span",
                 [RC.ROOT / "out" / "w_prep_v2" / "summary.json"])
    for l in hdr:
        print(l)
    print(f"# calibration files: {len(files)} under {REF}")
    cal = []
    for p in files:
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        if d.get("route") != "B" or d.get("wall_s") is None or not d.get("segments"):
            continue
        span = sum(abs(s.get("t_span") or 0.0) for s in d["segments"])
        cal.append({"file": str(Path(p).relative_to(REF)), "impl": d["impl"], "res": d["res"], "wall": d["wall_s"],
                    "span": span, "nseg": d.get("n_segments") or len(d["segments"]),
                    "laptop": "laptop" in p, "direct": (d.get("tmap") or {}).get("direct_segments")})
    rates = {}
    for impl in ("qawo", "fcc"):
        for res in (1, 2):
            C = [c for c in cal if c["impl"] == impl and c["res"] == res and not c["laptop"]]
            big = [c["wall"] / c["span"] * 1e6 for c in C if c["span"] >= 1e7]
            seg = [c["wall"] / c["nseg"] for c in C]
            rates[f"{impl}-r{res}"] = {"n": len(C), "n_span_ge_1e7": len(big),
                                       "us_per_rad_range": [min(big), max(big)] if big else None,
                                       "s_per_segment_range": [min(seg), max(seg)] if seg else None}
            print(f"{impl} r{res}: {len(C)} readings (Mac Studio); us/rad over spans >= 1e7: "
                  f"{rates[f'{impl}-r{res}']['us_per_rad_range']} ({len(big)}); s/segment {rates[f'{impl}-r{res}']['s_per_segment_range']}")
    for c in sorted(cal, key=lambda c: -c["span"])[:12]:
        print(f"   largest spans: {c['file']} span {c['span']:.3e} rad, {c['nseg']} segments, wall {c['wall']:.0f} s "
              f"({c['wall'] / c['span'] * 1e6:.4f} us/rad)")
    summ = json.loads((RC.ROOT / "out" / "w_prep_v2" / "summary.json").read_text())["calls"]
    wsam = json.loads((RC.ROOT / "out" / "sample" / "w_sample.json").read_text())
    proj = []
    for sid, s in summ.items():
        for impl in ("qawo", "fcc"):
            for res in (1, 2):
                r = rates[f"{impl}-r{res}"]
                per_seg = s["n_segments"] * r["s_per_segment_range"][1]
                per_rad = s["phase_sum_segments"] * r["us_per_rad_range"][1] * 1e-6 if r["us_per_rad_range"] else 0.0
                lo_rad = s["phase_sum_segments"] * r["us_per_rad_range"][0] * 1e-6 if r["us_per_rad_range"] else 0.0
                proj.append({"sid": sid, "unit": s["unit"], "index": s["index"], "impl": impl, "res": res,
                             "phase_span": s["phase_sum_segments"], "n_segments": s["n_segments"],
                             "projected_s": max(per_seg, per_rad), "projected_s_low": max(s["n_segments"] * r["s_per_segment_range"][0], lo_rad),
                             "per_segment_part_s": per_seg, "per_radian_part_s": per_rad})
    tot = sum(p["projected_s"] for p in proj)
    print(f"\nprojected total (sequential, six workers each): {tot:.0f} s = {tot / 3600:.2f} h")
    for p in sorted(proj, key=lambda p: -p["projected_s"])[:20]:
        print(f"   {p['sid']} {p['impl']} r{p['res']}: span {p['phase_span']:.3e} rad, {p['n_segments']} segments -> "
              f"{p['projected_s']:.0f} s (low {p['projected_s_low']:.0f} s)")
    (RC.ROOT / "out" / "cost_projection.json").write_text(json.dumps({"header": hdr, "calibration": cal, "rates": rates,
                                                                       "projection": proj, "total_s": tot}, indent=1))
    print(f"wrote out/cost_projection.json sha256 {RC.sha256(RC.ROOT / 'out' / 'cost_projection.json')}")


if __name__ == "__main__":
    main()
