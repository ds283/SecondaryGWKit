"""levin-phase-floor prompt 07b extension, section 2.7.4: project each new leaf reference (leafref_07b.py, FCC r1 and r2,
``--split-ratio 1.8``) by 07a's model before it is read. New in the extension, after Part A's leafproj_07b.py.

**The model** is 07a's refitted FCC fit (07a census out/refs/refs_projection.json, ``fcc_r1``/``fcc_r2``):
wall = alpha n_seg + beta span_f + gamma D_f, times its largest actual/fitted ratio as a margin. Part A's deviation 16:
**the gamma D_f term is kept for every segment the tool integrates directly**, D_f = h max|theta'| over that segment.

**The segments** are the ones the reading integrates: the production edges merged with the spec's points
(``merged_edges``), then, inside the spec's spans, ``split_steep`` at ratio 1.8 as leafref_07b.py does. Each segment is
classified by 07's per-segment rule (33 Chebyshev-Lobatto samples of theta'): **stationary** (theta' takes both signs:
the tool's direct path, whatever the split), **steep** (monotone with slope ratio above 2 after the split: direct too)
or **t-integrated**. D_f is summed over the stationary and steep segments, and reported apart for the stationary ones.
Part A's projection without the split (leafproj_07b.py's: every steep segment direct) is reported beside it as an upper
bound.

``--calibrate SPEC=READING`` reports a reading's actual wall and job CPU beside its projection (Part A's readings).
Writes OUT_JSON (refuses to overwrite).
Usage: PYTHONPATH=<tree>:<scripts> python -B leafproj_ext.py OUT_JSON SPEC [SPEC ...] [--calibrate SPEC=READING ...]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prov import REC07A  # noqa: E402
from leafref_07b import load_call, merged_edges, split_steep  # noqa: E402

FITS = REC07A / "out" / "refs" / "refs_projection.json"
SPLIT = 1.8
Y = np.cos(np.pi * np.arange(33) / 32)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def classify(thd, a, b):
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    d = np.array([float(thd(mid + half * v)) for v in Y])
    s = half * float(np.max(np.abs(d)))
    if not (np.all(d > 0) or np.all(d < 0)):
        return "stationary", s
    if float(np.max(np.abs(d)) / np.min(np.abs(d))) > 2.0:
        return "steep", s
    return "t", s


def features(call, edges, idx):
    th, thd = call["theta"]["theta"], call["theta"]["theta_deriv"]
    f = {"n_seg": 0, "span_f": 0.0, "D_f": 0.0, "D_f_stationary": 0.0, "n_stationary": 0, "n_steep": 0}
    for i in idx:
        a, b = edges[i], edges[i + 1]
        kind, s = classify(thd, a, b)
        f["n_seg"] += 1
        if kind == "t":
            f["span_f"] += abs(float(th(b)) - float(th(a)))
        else:
            f["D_f"] += s
            f["n_" + kind] += 1
            if kind == "stationary":
                f["D_f_stationary"] += s
    return f


def project(fits, f):
    out = {}
    for k in ("fcc_r1", "fcc_r2"):
        al, be, ga = fits[k]["alpha_beta_gamma"]
        w = al * f["n_seg"] + be * f["span_f"] + ga * f["D_f"]
        out[k] = {"fit_s": w, "with_margin_s": w * max(1.0, fits[k]["max_actual_over_fit"]),
                  "gamma_D_f_s": ga * f["D_f"], "gamma_D_f_stationary_s": ga * f["D_f_stationary"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--calibrate", nargs="*", default=[])
    a = ap.parse_args()
    out_p = Path(a.out_json)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    import tools.filon_reference as F

    fits = json.loads(FITS.read_text())["fits"]
    cal = dict(x.split("=") for x in a.calibrate)
    res, total, total_unsplit = {}, {"fcc_r1": 0.0, "fcc_r2": 0.0}, {"fcc_r1": 0.0, "fcc_r2": 0.0}
    for sp in list(a.specs) + [s for s in cal if s not in a.specs]:
        spec = json.loads(Path(sp).read_text())
        c = spec["capture"]
        ov = c.get("override") or {}
        call = load_call(c["path"], c["offset"], c["length"], c["sha256"], ov.get("rtol"), ov.get("atol"))
        fl = float.fromhex
        points = sorted({fl(p) for p in spec["points"]})
        spans = [(fl(s0), fl(s1)) for s0, s1 in spec["spans"]]
        prod, _ = F.production_edges(call)
        edges0 = merged_edges(prod, points)

        def inside(edges):
            ix = set()
            for s0, s1 in spans:
                ix |= set(range(edges.index(s0), edges.index(s1)))
            return sorted(ix)

        f0 = features(call, edges0, inside(edges0))
        add = []
        for s0, s1 in spans:
            add += split_steep(call, edges0, s0, s1, SPLIT)
        edges = sorted(set(edges0) | set(add))
        f = features(call, edges, inside(edges))
        proj, proj0 = project(fits, f), project(fits, f0)
        counted = sp in a.specs
        for k in total:
            if counted:
                total[k] += proj[k]["with_margin_s"]
                total_unsplit[k] += proj0[k]["with_margin_s"]
        row = {"spec": sp, "spec_sha256": sha256(sp), "counted": counted, "features_split": f, "features_unsplit": f0,
               "projection_split": proj, "projection_unsplit": proj0}
        if sp in cal:
            r = json.loads(Path(cal[sp]).read_text())
            row["actual"] = {f"fcc_r{k}": {"wall_s": v["wall_s"], "job_cpu_s": sum(v["job_wall_s"])}
                             for k, v in r["readings"].items()}
            row["reading"], row["reading_sha256"], row["reading_split_ratio"] = cal[sp], sha256(cal[sp]), r.get("split_ratio")
        res[f"{spec['sid']}:{Path(sp).stem}"] = row
        act = row.get("actual")
        print(f"{spec['sid']:16s} segs {f['n_seg']:5d} (stationary {f['n_stationary']}, steep {f['n_steep']}, "
              f"D_f {f['D_f']:.3g}, of it stationary {f['D_f_stationary']:.3g}) -> r1 {proj['fcc_r1']['with_margin_s']:.0f} s "
              f"r2 {proj['fcc_r2']['with_margin_s']:.0f} s [gamma D_f {proj['fcc_r1']['gamma_D_f_s'] + proj['fcc_r2']['gamma_D_f_s']:.0f} s]; "
              f"unsplit r1+r2 {proj0['fcc_r1']['with_margin_s'] + proj0['fcc_r2']['with_margin_s']:.0f} s"
              + (f"; ACTUAL job CPU r1 {act['fcc_r1']['job_cpu_s']:.0f} s r2 {act['fcc_r2']['job_cpu_s']:.0f} s, wall "
                 f"{act['fcc_r1']['wall_s'] + act['fcc_r2']['wall_s']:.0f} s" if act else "")
              + ("" if counted else "  (calibration only)"), flush=True)
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "fits": str(FITS),
                      "fits_sha256": sha256(FITS), "split_ratio": SPLIT},
           "fits": {k: fits[k] for k in ("fcc_r1", "fcc_r2")}, "calls": res, "total_with_margin_s": total,
           "total_unsplit_with_margin_s": total_unsplit}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(f"TOTAL (job CPU, with margin): r1 {total['fcc_r1']:.0f} s, r2 {total['fcc_r2']:.0f} s, "
          f"{sum(total.values()) / 3600:.3f} h; unsplit {sum(total_unsplit.values()) / 3600:.3f} h")
    print("wrote", out_p, sha256(out_p))


if __name__ == "__main__":
    main()
