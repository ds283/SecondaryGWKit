"""levin-phase-floor prompt 07a section 6, the user's references: project each reading's wall time before any is taken.
New in 07a, after 07's refs_projection_07.py (a record, not edited), whose features (``features``, imported from it,
unchanged) and model form it keeps. 07's calibration (06g's readings on this host, P1's, the laptop's) is no longer all
on this host, so the model is fitted instead to 07's own 74 V4 readings, taken on this host with the same tool
(tools/filon_reference.py d37e633b..., byte-identical in h) and the same driver settings:
- per reading kind (impl, res): wall = alpha n_seg + beta span + gamma D, by non-negative least squares over the readings
  of that kind among 07's 24 calls (QAWO: span_q, D_q; FCC: span_f, D_f), each call's features from its 07 capture
  (var/levin-phase-floor-07/census/state/v4/inputs/<sid>.pkl, read only);
- the projection of each of the four reference calls (out/refs/inputs.json), with the fit's largest ratio of actual to
  fitted wall over 07's readings as a margin;
- 07's ruling at its stop 4 decides which readings are taken: FCC r1 and r2 on every call, QAWO r1 and r2 on a call whose
  QAWO readings each project at 30 minutes or less (with the margin).
Writes OUT_JSON. Usage: PYTHONPATH=<tree h>:<scripts> python -B refs_projection_07a.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07, WIDER, header, sha256  # noqa: E402
from refs_projection_07 import features, load_capture  # noqa: E402  (07's, unchanged)

V4IN = REC07 / "out" / "v4" / "inputs.json"
READ = REC07 / "out" / "v4" / "readings"
MINE = WIDER / "out" / "refs" / "inputs.json"
KINDS = [("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2)]


def nnls(A, y):
    from scipy.optimize import nnls as _n

    return _n(A, y)[0]


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: the user's references, projected", [V4IN, MINE])
    v4 = json.loads(V4IN.read_text())["inputs"]
    feat07 = {}
    for sid, e in sorted(v4.items()):
        if sha256(e["path"]) != e["sha256"]:
            raise SystemExit(f"{sid}: 07's capture has another SHA-256")
        feat07[sid] = features(load_capture(e["path"]))
    fits = {}
    for impl, res in KINDS:
        X, y, names = [], [], []
        for sid in sorted(v4):
            p = READ / f"{sid}-{impl}-r{res}.json"
            if not p.exists():
                continue
            f = feat07[sid]
            X.append([f["n_seg"], f["span_q" if impl == "qawo" else "span_f"], f["D_q" if impl == "qawo" else "D_f"]])
            y.append(json.loads(p.read_text())["wall_s"])
            names.append(sid)
        A, Y = np.array(X, float), np.array(y, float)
        coef = nnls(A, Y)
        fit = A @ coef
        ratio = Y / np.where(fit > 0, fit, np.nan)
        fits[f"{impl}_r{res}"] = {"n": len(y), "alpha_beta_gamma": coef.tolist(), "max_actual_over_fit": float(np.nanmax(ratio)),
                                  "min_actual_over_fit": float(np.nanmin(ratio)), "readings": dict(zip(names, y))}
    mine = json.loads(MINE.read_text())["inputs"]
    calls = {}
    for sid, e in sorted(mine.items()):
        if sha256(e["path"]) != e["sha256"]:
            raise SystemExit(f"{sid}: capture SHA-256 differs")
        f = features(load_capture(e["path"]))
        proj = {}
        for impl, res in KINDS:
            k = f"{impl}_r{res}"
            a, b, g = fits[k]["alpha_beta_gamma"]
            w = a * f["n_seg"] + b * f["span_q" if impl == "qawo" else "span_f"] + g * f["D_q" if impl == "qawo" else "D_f"]
            proj[k] = {"fit_s": w, "with_margin_s": w * max(1.0, fits[k]["max_actual_over_fit"])}
        qawo_ok = all(proj[f"qawo_r{r}"]["with_margin_s"] <= 1800.0 for r in (1, 2))
        calls[sid] = {"features": f, "projection": proj, "qawo": qawo_ok,
                      "readings": ([("qawo", 1), ("qawo", 2)] if qawo_ok else []) + [("fcc", 1), ("fcc", 2)]}
    total = sum(calls[s]["projection"][f"{i}_r{r}"]["with_margin_s"] for s in calls for i, r in calls[s]["readings"])
    doc = {"header": hdr, "fits": fits, "calls": calls, "total_with_margin_s": total}
    out_p.write_text(json.dumps(doc, indent=1))
    for k, v in fits.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "readings"})
    for s, c in calls.items():
        print(s, c["features"], {k: round(v["with_margin_s"]) for k, v in c["projection"].items()}, "QAWO" if c["qawo"] else "FCC only")
    print(f"total, one reading at a time, with margin: {total:.0f} s")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
