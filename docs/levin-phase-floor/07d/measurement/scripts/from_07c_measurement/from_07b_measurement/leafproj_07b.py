"""levin-phase-floor prompt 07b Part A, section 2.2: project each piece reading's wall (leafref_07b.py, FCC r1 and r2) before
it is taken. New in 07b.

Features as 07's ``features`` (refs_projection_07.py, its per-segment rule copied here unchanged in substance), but on
the reading's own edges (production edges and the spec's points merged as leafref_07b.py merges them) and only over the
segments inside the spec's spans: n_seg, span_f (the t-integrated span), D_f (sum over the directly integrated segments
of h max|theta'|). The model is 07a's refitted FCC fit (07a census out/refs/refs_projection.json, ``fcc_r1``/``fcc_r2``:
alpha n_seg + beta span_f + gamma D_f), with its largest actual/fitted ratio as a margin. ``--calibrate READING_JSON``
(a leafref output) reports the actual job CPU and wall beside the projection. Writes OUT_JSON.
Usage: PYTHONPATH=<tree>:<scripts> python -B leafproj_07b.py OUT_JSON SPEC [SPEC ...] [--calibrate SPEC=READING ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07A, header, sha256  # noqa: E402
from leafref_07b import load_call, merged_edges  # noqa: E402

FITS = REC07A / "out" / "refs" / "refs_projection.json"


def seg_features(call, edges):
    th, thd = call["theta"]["theta"], call["theta"]["theta_deriv"]
    y = np.cos(np.pi * np.arange(33) / 32)
    out = {"n_seg": 0, "span_f": 0.0, "D_f": 0.0, "n_direct": 0}
    for a, b in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        sgn = np.array([float(thd(mid + half * v)) for v in y])
        d = np.abs(sgn)
        s = half * float(np.max(d))
        out["n_seg"] += 1
        mono = bool(np.all(sgn > 0) or np.all(sgn < 0))
        if not mono or float(np.max(d) / np.min(d)) > 2.0:
            out["D_f"] += s
            out["n_direct"] += 1
        else:
            out["span_f"] += abs(float(th(b)) - float(th(a)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--calibrate", nargs="*", default=[])
    a = ap.parse_args()
    import tools.filon_reference as F

    fits = json.loads(FITS.read_text())["fits"]
    cal = dict(x.split("=") for x in a.calibrate)
    res = {}
    total = {"fcc_r1": 0.0, "fcc_r2": 0.0}
    for sp in a.specs:
        spec = json.loads(Path(sp).read_text())
        c = spec["capture"]
        ov = c.get("override") or {}
        call = load_call(c["path"], c["offset"], c["length"], c["sha256"], ov.get("rtol"), ov.get("atol"))
        fl = float.fromhex
        points = sorted({fl(p) for p in spec["points"]})
        prod, _ = F.production_edges(call)
        edges = merged_edges(prod, points)
        inside = set()
        for s0, s1 in spec["spans"]:
            i0, i1 = edges.index(fl(s0)), edges.index(fl(s1))
            inside |= set(range(i0, i1))
        segs = sorted(inside)
        f = {"n_seg": 0, "span_f": 0.0, "D_f": 0.0, "n_direct": 0}
        for i in segs:
            g = seg_features(call, [edges[i], edges[i + 1]])
            for k in f:
                f[k] += g[k]
        proj = {}
        for k in ("fcc_r1", "fcc_r2"):
            al, be, ga = fits[k]["alpha_beta_gamma"]
            w = al * f["n_seg"] + be * f["span_f"] + ga * f["D_f"]
            proj[k] = {"fit_s": w, "with_margin_s": w * max(1.0, fits[k]["max_actual_over_fit"])}
            total[k] += proj[k]["with_margin_s"]
        row = {"spec": sp, "spec_sha256": sha256(sp), "features": f, "projection": proj}
        if sp in cal:
            r = json.loads(Path(cal[sp]).read_text())
            row["actual"] = {f"fcc_r{k}": {"wall_s": v["wall_s"], "job_cpu_s": sum(v["job_wall_s"]),
                                           "max_job_s": max(v["job_wall_s"])} for k, v in r["readings"].items()}
        res[spec["sid"] + ":" + Path(sp).stem] = row
        print(f"{Path(sp).stem}: {f} -> r1 {proj['fcc_r1']['with_margin_s']:.0f} s, r2 {proj['fcc_r2']['with_margin_s']:.0f} s"
              + (f"; actual {row['actual']}" if "actual" in row else ""), flush=True)
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: section 2.2's readings, projected", [FITS])
    Path(a.out_json).write_text(json.dumps({"header": hdr, "fits": {k: fits[k] for k in ("fcc_r1", "fcc_r2")}, "calls": res,
                                            "total_with_margin_s": total}, indent=1))
    print(f"total with margin: r1 {total['fcc_r1']:.0f} s, r2 {total['fcc_r2']:.0f} s, both {sum(total.values()) / 3600:.2f} h "
          "(one reading at a time, six workers)")


if __name__ == "__main__":
    main()
