"""levin-phase-floor prompt 07 section 6, V4: project the references' wall time (four readings of each of the 24
captured calls by tools/filon_reference.py at g, one reading at a time, six workers), from earlier readings of the same
tool's code on this host, before any reading is taken. New in 07. Read only (captures are unpickled; no reading, no
Ray, no store).

**The features of a captured Route B call** (``features``): the tool's own segment edges
(``tools.filon_reference.production_edges``); on each segment theta' (the call's ``theta_deriv``) at 33
Chebyshev-Lobatto points: monotone if of one strict sign, its slope ratio max|theta'|/min|theta'|, its half width h and
its phase extent at the largest slope, s = h max|theta'| (the tool's direct panels number 2.5 s / dphi); and its phase
span, |theta(X_r) - theta(X_l)| for a monotone segment, the sum of |Delta theta| over the samples otherwise. Then:
n_seg; span_mono, the span over segments the reading integrates in t; D_q = sum of s over the non-monotone segments
(QAWO's direct segments); D_f = sum of s over the non-monotone segments and those with slope ratio above 2 (FCC's).

**The model, per reading (impl, res):** wall = alpha n_seg + beta span_mono + gamma D (D_q or D_f).
- alpha and beta by least squares (non-negative) on 06g's readings at this host (var/levin-phase-floor-06g/references/
  out/readings/, the tools at 1c5336e and ff3e8394) of every Route B call but the stationary-point samples, whose
  reading has no direct or non-monotone segment, with n_seg and span from the reading's own segment table (06g's
  captures are no longer on this host); for the 24, n_seg and span come from ``features``, checked against z12's and
  P1's rows' own tables (``feature_check``);
- gamma from the stationary-point calls read with the tool's present code: P1's two rows (this host, the tool at
  74600350, byte-identical to g's), and z00 and z12 (QAWO: 06g's rerun batch at ff3e8394 on this host; FCC: the laptop's
  batch at b5626ecb, scaled to this host by r55's FCC readings on both hosts), their features from this census's
  calibration capture of z12 (z00's integrand); gamma = the largest over these of (wall - alpha n_seg - beta span_mono)/D.
**The projection:** the sum over the 24 calls and the four readings, one reading at a time; and the same multiplied by
1.5 for the load and the model's scatter. A projection of a day or more is a stop (prompt section 6).

Usage: PYTHONPATH=<tree g>:<scripts> python -B refs_projection_07.py OUT_JSON
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import WIDER, header, sha256  # noqa: E402

R06G = RC.MAIN / "var" / "levin-phase-floor-06g" / "references"
LAPTOP = RC.G_TREE / "docs" / "levin-phase-floor" / "06g" / "references" / "out" / "readings_fcc_b5626ec_laptop"
P1 = RC.MAIN / "var" / "levin-phase-floor-07" / "rehearsal" / "out" / "p1_references"
CAPS = WIDER / "out" / "v4" / "captures.json"
KINDS = [("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2)]


def load_capture(path, offset=0, length=None, sha=None):
    import hashlib

    import ray.cloudpickle as cp

    with open(path, "rb") as f:
        f.seek(offset)
        blob = f.read() if length is None else f.read(length)
    if sha is not None and hashlib.sha256(blob).hexdigest() != sha:
        raise RuntimeError(f"{path}: sha256 differs")
    return cp.loads(blob)


def features(call):
    from tools.filon_reference import production_edges

    edges, _ = production_edges(call)
    th, thd = call["theta"]["theta"], call["theta"]["theta_deriv"]
    y = np.cos(np.pi * np.arange(33) / 32)
    out = {"n_seg": len(edges) - 1, "span_q": 0.0, "span_f": 0.0, "D_q": 0.0, "D_f": 0.0, "n_nonmono": 0, "n_steep": 0}
    for a, b in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        d = np.array([float(thd(mid + half * v)) for v in y])
        mono = bool(np.all(d > 0) or np.all(d < 0))
        s = half * float(np.max(np.abs(d)))
        span = abs(float(th(b)) - float(th(a)))
        if not mono:
            out["D_q"] += s
            out["D_f"] += s
            out["n_nonmono"] += 1
            continue
        out["span_q"] += span
        if float(np.max(np.abs(d)) / np.min(np.abs(d))) > 2.0:
            out["D_f"] += s
            out["n_steep"] += 1
        else:
            out["span_f"] += span
    return out


def table_feats(r):
    """A reading's own segment table: its segment count and the |t_span| summed over the segments it integrates in t
    (those not ``direct``)."""
    segs = r["segments"]
    return {"n_seg": len(segs), "span": sum(abs(x["t_span"]) for x in segs if not x.get("direct")),
            "n_direct": sum(1 for x in segs if x.get("direct"))}


def reading_direct(r):
    return sum(1 for s in r["segments"] if s.get("direct")) if isinstance(r.get("segments"), list) else None


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: V4's references' wall projection",
                 [CAPS, R06G / "out" / "capture.json"])
    doc = {"header": hdr, "model": __doc__}
    # 1. alpha, beta from 06g's Route B readings with no direct segment
    rows = {k: [] for k in KINDS}
    for p in sorted((R06G / "out" / "readings").glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("route") != "B" or r.get("wall_s") is None or not isinstance(r.get("segments"), list):
            continue
        if r["sample"][0] in ("z", "w") or any(x.get("direct") or x.get("monotone") is False for x in r["segments"]):
            continue  # the stationary-point samples go to gamma; any reading with a direct segment is left out
        t = table_feats(r)
        rows[(r["impl"], r["res"])].append((p.name, t["n_seg"], t["span"], r["wall_s"]))
    coef = {}
    for k, rs in rows.items():
        A = np.array([[n, s] for _, n, s, _ in rs], dtype=float)
        b = np.array([w for *_, w in rs], dtype=float)
        from scipy.optimize import nnls

        x, _ = nnls(A, b)
        pred = A @ x
        coef[k] = {"alpha": float(x[0]), "beta": float(x[1]), "n": len(rs),
                   "max_ratio_obs_over_pred": float(np.max(b / np.maximum(pred, 1e-9))) if len(rs) else None,
                   "median_ratio": float(np.median(b / np.maximum(pred, 1e-9))) if len(rs) else None}
    doc["alpha_beta"] = {f"{k[0]}_r{k[1]}": v for k, v in coef.items()}
    # 2. gamma from the stationary-point calls read with the tool's present code
    caps07 = {e["sid"]: e for e in json.loads(CAPS.read_text())}
    cz = caps07["cal_z12"]
    fz = features(load_capture(Path(cz["dir"]) / cz["pickle"], cz["offset"], cz["length"], cz["sha256"]))
    host = {}
    for res in (1, 2):
        mac = json.loads((R06G / "out" / "readings" / f"r55-fcc-r{res}.json").read_text())["wall_s"]
        lap = json.loads((LAPTOP / f"r55-fcc-r{res}.json").read_text())["wall_s"]
        host[res] = mac / lap
    doc["host_factor_mac_over_laptop_fcc"] = host
    gam = {k: [] for k in KINDS}
    for sid in ("z00", "z12"):
        for impl, res in KINDS:
            if impl == "qawo":
                r = json.loads((R06G / "out" / "readings" / f"{sid}-qawo-r{res}-rerun.json").read_text())
                w = r["wall_s"]
            else:
                r = json.loads((LAPTOP / f"{sid}-fcc-r{res}.json").read_text())
                w = r["wall_s"] * host[res]
            gam[(impl, res)].append((sid, fz, table_feats(r), w))
    p1calls = json.loads((P1 / "calls.json").read_text())
    doc["p1_features"] = {}
    for c in p1calls:
        f = features(load_capture(c["pickle"], sha=c["pickle_sha256"]))
        doc["p1_features"][f"row{c['p1']}"] = f
        rd = json.loads(Path(c["out"]).read_text())["readings"]
        for r in rd:
            gam[(r["impl"], r["res"])].append((f"p1_row{c['p1']}", f, table_feats(r), r["wall_s"]))
    doc["z12_features"] = fz
    # the features' check against the readings' own tables (z12, P1's rows): segments and spans
    doc["feature_check"] = [{"sample": sid, "reading": f"{k[0]}_r{k[1]}", "n_seg": [f["n_seg"], t["n_seg"]],
                             "span": [f["span_q"] if k[0] == "qawo" else f["span_f"], t["span"]],
                             "n_direct": [f["n_nonmono"] if k[0] == "qawo" else f["n_nonmono"] + f["n_steep"], t["n_direct"]]}
                            for k, lst in gam.items() for sid, f, t, _ in lst]
    gamma = {}
    for k, lst in gam.items():
        cf = coef[k]
        vals = []
        for sid, f, t, w in lst:
            D = f["D_q"] if k[0] == "qawo" else f["D_f"]
            rest = w - cf["alpha"] * t["n_seg"] - cf["beta"] * t["span"]
            vals.append({"sample": sid, "wall_s": w, "D": D, "gamma": rest / D if D > 0 else None})
        gamma[k] = {"samples": vals, "gamma": max(v["gamma"] for v in vals if v["gamma"] is not None)}
    doc["gamma"] = {f"{k[0]}_r{k[1]}": v for k, v in gamma.items()}
    # 3. the 24
    proj = []
    for sid, e in sorted(caps07.items()):
        if sid == "cal_z12":
            continue
        f = features(load_capture(Path(e["dir"]) / e["pickle"], e["offset"], e["length"], e["sha256"]))
        per = {}
        for k in KINDS:
            D = f["D_q"] if k[0] == "qawo" else f["D_f"]
            span = f["span_q"] if k[0] == "qawo" else f["span_f"]
            per[f"{k[0]}_r{k[1]}"] = coef[k]["alpha"] * f["n_seg"] + coef[k]["beta"] * span + max(gamma[k]["gamma"], 0.0) * D
        proj.append({"sid": sid, "unit": e["unit"], "features": f, "readings_s": per, "total_s": sum(per.values())})
    total = sum(p["total_s"] for p in proj)
    doc["calls"] = proj
    doc["total_s"] = total
    doc["total_with_1p5_s"] = 1.5 * total
    doc["largest_reading_s"] = max((v, p["sid"], k) for p in proj for k, v in p["readings_s"].items())
    doc["n_readings_over_1800_s"] = sum(1 for p in proj for v in p["readings_s"].values() if v > 1800)
    doc["beyond_a_day"] = 1.5 * total >= 86400 or total >= 86400
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: doc[k] for k in ("alpha_beta", "host_factor_mac_over_laptop_fcc", "z12_features", "p1_features", "feature_check")}, indent=1))
    for k, v in doc["gamma"].items():
        print(k, "gamma", v["gamma"], [(s["sample"], round(s["wall_s"]), round(s["D"]), s["gamma"]) for s in v["samples"]])
    for p in proj:
        print(p["sid"], p["unit"], {k: round(v) for k, v in p["readings_s"].items()}, round(p["total_s"]), p["features"])
    print(f"total {total:.0f} s ({total / 3600:.2f} h); x1.5 {1.5 * total:.0f} s; largest reading {doc['largest_reading_s']}; "
          f"readings over 30 min {doc['n_readings_over_1800_s']}; a day or more: {doc['beyond_a_day']}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
