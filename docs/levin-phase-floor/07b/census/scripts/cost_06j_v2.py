"""levin-phase-floor prompt 06j, Part W: the readings' cost from phase span, second form. New in 06j; supersedes
cost_06j.py's projection (out/cmd/47), whose "largest observed rate" took the per-radian rate from readings whose
wall is mostly per-segment overhead (span just above 1e7 rad, a few thousand segments) and so projected FCC and QAWO
at 10-100x their observed cost on the largest spans; cost_06j.py's output is kept as it was.

Model: for each (impl, res), wall = a * n_segments + b * phase_span, fitted by non-negative least squares
(scipy.optimize.nnls) on 06g's Mac Studio Route B readings that finished (the same calibration set as
cost_06j.py: readings/, rereads/; each reading's phase span the sum of its segments' ``t_span``), with each
reading's ratio observed/fitted. The projection for a W reading is the fit times the **largest** ratio observed in
the calibration (an upper envelope), and the fit alone is reported beside it. b is QAWO's per-radian rate
(06g: 0.057-0.10 us/rad at resolution 1 on six workers) and should be near zero for FCC.

Usage: python -B cost_06j_v2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402


def main():
    v1 = json.loads((RC.ROOT / "out" / "cost_projection.json").read_text())
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: Part W's reading costs from phase span (two-term fit)",
                 [RC.ROOT / "out" / "cost_projection.json", RC.ROOT / "out" / "w_prep_v2" / "summary.json"])
    for l in hdr:
        print(l)
    cal = [c for c in v1["calibration"] if not c["laptop"]]
    fits = {}
    for impl in ("qawo", "fcc"):
        for res in (1, 2):
            C = [c for c in cal if c["impl"] == impl and c["res"] == res]
            A = np.array([[c["nseg"], c["span"]] for c in C], dtype=float)
            y = np.array([c["wall"] for c in C])
            coef, _ = nnls(A, y)
            pred = A @ coef
            ratio = y / np.maximum(pred, 1e-9)
            big = [c for c in C if c["span"] >= 1e10]
            fits[f"{impl}-r{res}"] = {"n": len(C), "s_per_segment": coef[0], "us_per_rad": coef[1] * 1e6,
                                      "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
                                      "ratio_median": float(np.median(ratio)),
                                      "largest_spans": [(c["file"], c["span"], c["nseg"], c["wall"]) for c in big]}
            print(f"{impl} r{res}: n {len(C)}; wall = {coef[0]:.4g} s/segment x n + {coef[1] * 1e6:.4g} us/rad x span; "
                  f"observed/fitted min {ratio.min():.3g} median {np.median(ratio):.3g} max {ratio.max():.3g}")
            for c, rr in sorted(zip(C, ratio), key=lambda t: -t[1])[:3]:
                print(f"     worst ratio {rr:.3g}: {c['file']} span {c['span']:.3e} nseg {c['nseg']} wall {c['wall']:.1f} s")
    summ = json.loads((RC.ROOT / "out" / "w_prep_v2" / "summary.json").read_text())["calls"]
    proj = []
    for sid, s in summ.items():
        for impl in ("qawo", "fcc"):
            for res in (1, 2):
                f = fits[f"{impl}-r{res}"]
                fit = f["s_per_segment"] * s["n_segments"] + f["us_per_rad"] * 1e-6 * s["phase_sum_segments"]
                proj.append({"sid": sid, "unit": s["unit"], "index": s["index"], "impl": impl, "res": res,
                             "phase_span": s["phase_sum_segments"], "n_segments": s["n_segments"], "fit_s": fit,
                             "projected_s": fit * max(1.0, f["ratio_max"])})
    tot = sum(p["projected_s"] for p in proj)
    totf = sum(p["fit_s"] for p in proj)
    print(f"\nprojected total, readings one after another: fit {totf:.0f} s ({totf / 3600:.2f} h); envelope {tot:.0f} s ({tot / 3600:.2f} h)")
    for p in sorted(proj, key=lambda p: -p["projected_s"]):
        print(f"   {p['sid']} {p['unit']} #{p['index']} {p['impl']} r{p['res']}: span {p['phase_span']:.4g} rad, {p['n_segments']} segments: "
              f"fit {p['fit_s']:.0f} s, envelope {p['projected_s']:.0f} s")
    (RC.ROOT / "out" / "cost_projection_v2.json").write_text(json.dumps({"header": hdr, "fits": fits, "projection": proj,
                                                                          "total_fit_s": totf, "total_envelope_s": tot}, indent=1))
    print(f"wrote out/cost_projection_v2.json sha256 {RC.sha256(RC.ROOT / 'out' / 'cost_projection_v2.json')}")


if __name__ == "__main__":
    main()
