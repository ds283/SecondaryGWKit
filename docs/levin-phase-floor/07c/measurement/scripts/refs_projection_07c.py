"""levin-phase-floor prompt 07c Part A, section 2.5: project R4's and R5's reference readings before any is taken. New in 07c.

**The readings** (07's V4 method, as 07b Part A read Route B): split FCC (leafref_07c.py, ``--split-ratio 1.8``) at
resolutions 1 and 2, and QAWO (one_reading.py) at 1 and 2 where each projects at 30 minutes or less with its margin.

**The model** is 07b's, with its gamma D term for every segment holding a stationary point (07b Part A's deviation 16):
- features of each integrand by 07's ``features`` (refs_projection_07.py, imported unchanged): its production segments,
  and D_nonmono, the summed half-width x max|theta'| of the segments on which theta' changes sign (a stationary point,
  integrated directly by the tool whatever the split); and n_split, the edges after leafref's ``split_steep`` at 1.8;
- **split FCC, per resolution:** wall = a_r n_split + g_r D_nonmono, fitted by non-negative least squares on 07b Part A's
  own split-FCC readings (var/levin-phase-floor-07b/measurement/out/refs/readings/*-fccsplit.json, wall per resolution
  at two workers, each input's features recomputed from its kept input pickle), with the fit's largest actual/fitted
  ratio as its margin;
- **QAWO:** 07a's refitted model (07a census out/refs/refs_projection.json ``fits``, alpha n_seg + beta span_q + gamma D_q)
  with its margin, as 07b used it;
- **the run:** three readings at a time, two workers each (07b's plan), so the elapsed projection is the summed wall over
  three, and the whole is multiplied by 1.20 (07b's census projection ran 1.20x short; the brief).

Reads out/refs/inputs.json. Writes out/refs/refs_projection.json. Usage: PYTHONPATH=<tree>:<scripts> python -B
refs_projection_07c.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, REC07A, WIDER, header, sha256  # noqa: E402
from refs_projection_07 import features, load_capture  # noqa: E402  (07's, unchanged)
from leafref_07c import split_steep  # noqa: E402

FITS = REC07A / "out" / "refs" / "refs_projection.json"
R07B = MAIN / "var/levin-phase-floor-07b/measurement"
SPLIT = 1.8
CENSUS_MARGIN = 1.20
QAWO_CAP_S = 1800.0
PARALLEL = 3


def feats(call):
    from tools.filon_reference import production_edges

    f = features(call)
    edges, _ = production_edges(call)
    lo, hi = min(edges), max(edges)
    f["n_split"] = len(edges) - 1 + len(split_steep(call, list(edges), lo, hi, SPLIT))
    th, thd = call["theta"]["theta"], call["theta"]["theta_deriv"]
    y = np.cos(np.pi * np.arange(33) / 32)
    dn = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        d = np.array([float(thd(mid + half * v)) for v in y])
        if not (np.all(d > 0) or np.all(d < 0)):
            dn += half * float(np.max(np.abs(d)))
    f["D_nonmono"] = dn
    return f


def main():
    out = WIDER / "out" / "refs" / "refs_projection.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    from scipy.optimize import nnls

    # the split-FCC fit, on 07b Part A's readings
    ins07b = json.loads((R07B / "out" / "refs" / "inputs.json").read_text())["inputs"]
    rows = {1: [], 2: []}
    calib = {}
    for p in sorted((R07B / "out" / "refs" / "readings").glob("*-fccsplit.json")):
        d = json.loads(p.read_text())
        sid = d["sid"]
        e = ins07b[sid]
        f = feats(load_capture(e["path"], sha=e["sha256"]))
        calib[sid] = {"features": f, "wall": {r: d["readings"][str(r)]["wall_s"] for r in (1, 2) if str(r) in d["readings"]}}
        for r in (1, 2):
            if str(r) in d["readings"]:
                rows[r].append((f["n_split"], f["D_nonmono"], d["readings"][str(r)]["wall_s"]))
    fit = {}
    for r, rs in rows.items():
        A = np.array([[n, D] for n, D, _ in rs], float)
        b = np.array([w for *_, w in rs], float)
        x, _ = nnls(A, b)
        pred = A @ x
        ratio = b / np.maximum(pred, 1e-9)
        fit[r] = {"a_g": x.tolist(), "n": len(rs), "max_actual_over_fit": float(ratio.max()),
                  "min_actual_over_fit": float(ratio.min()), "median_actual_over_fit": float(np.median(ratio))}
    qfits = json.loads(FITS.read_text())["fits"]
    ins = json.loads((WIDER / "out" / "refs" / "inputs.json").read_text())["inputs"]
    calls = {}
    for key, e in sorted(ins.items()):
        call = load_capture(e["path"], sha=e["sha256"])
        f = feats(call)
        proj = {}
        for r in (1, 2):
            a_, g_ = fit[r]["a_g"]
            w = a_ * f["n_split"] + g_ * f["D_nonmono"]
            proj[f"fccsplit_r{r}"] = {"fit_s": w, "with_margin_s": w * max(1.0, fit[r]["max_actual_over_fit"])}
            al, be, ga = qfits[f"qawo_r{r}"]["alpha_beta_gamma"]
            w = al * f["n_seg"] + be * f["span_q"] + ga * f["D_q"]
            proj[f"qawo_r{r}"] = {"fit_s": w, "with_margin_s": w * max(1.0, qfits[f"qawo_r{r}"]["max_actual_over_fit"])}
        qawo = all(proj[f"qawo_r{r}"]["with_margin_s"] <= QAWO_CAP_S for r in (1, 2))
        readings = ["fccsplit_r1", "fccsplit_r2"] + (["qawo_r1", "qawo_r2"] if qawo else [])
        calls[key] = {"draw": e["draw"], "features": f, "projection": proj, "qawo": qawo, "readings": readings,
                      "wall_s": sum(proj[k]["with_margin_s"] for k in readings),
                      "longest_reading_s": max(proj[k]["with_margin_s"] for k in readings)}
    job = sum(c["wall_s"] for c in calls.values())
    longest = max(c["longest_reading_s"] for c in calls.values())
    elapsed = max(job / PARALLEL, longest) * CENSUS_MARGIN
    by_draw = {}
    for c in calls.values():
        by_draw[c["draw"]] = by_draw.get(c["draw"], 0.0) + c["wall_s"]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c Part A: R4 and R5, projected", [FITS, WIDER / "out" / "refs" / "inputs.json"])
    doc = {"header": hdr, "split_fit": fit, "qawo_fits": qfits, "calibration": calib, "calls": calls,
           "job_wall_s": job, "by_draw_job_wall_s": by_draw, "longest_reading_s": longest, "parallel": PARALLEL,
           "census_margin": CENSUS_MARGIN, "elapsed_projection_s": elapsed}
    out.write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    print("split-FCC fit:", json.dumps(fit))
    for k, c in sorted(calls.items(), key=lambda kv: -kv[1]["wall_s"]):
        print(f"{k:28s} n_seg {c['features']['n_seg']:4d} n_split {c['features']['n_split']:5d} D_nonmono {c['features']['D_nonmono']:.3g} "
              f"qawo {'yes' if c['qawo'] else 'no '} " + " ".join(f"{r} {c['projection'][r]['with_margin_s']:.0f}" for r in c["readings"])
              + f" | {c['wall_s']:.0f} s")
    print(f"job wall {job:.0f} s ({job / 3600:.2f} h); by draw {({k: round(v) for k, v in by_draw.items()})}; longest reading {longest:.0f} s")
    print(f"elapsed, {PARALLEL} at once, x{CENSUS_MARGIN}: {elapsed:.0f} s = {elapsed / 3600:.2f} h")
    print(f"wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
