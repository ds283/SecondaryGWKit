"""levin-phase-floor 07b section 5 (Part C), the R31 rehearsal agent: the wall projection of the references for the group
records of interest (the user's word at stop 1, "settle which store understates"), before any reading is taken. New, after
Part A's refs_projection_07b.py (a record, not edited). Read only: the captures are unpickled (ray.cloudpickle; Ray never
initialised); nothing is read by the tool.

- ``load_capture`` and ``features`` are copied verbatim from 07's refs_projection_07.py (a record): the tool's own
  production edges (``tools.filon_reference.production_edges``), theta' at 33 Chebyshev-Lobatto points per segment, n_seg,
  span_q / span_f, D_q / D_f (the non-monotone segments, and for FCC those of slope ratio above 2), n_nonmono, n_steep.
- **07a's refitted model, as the prompt specifies** (07a census out/refs/refs_projection.json, ``fits``; Part A's
  deviation 16: the gamma D term is kept): per kind, wall = alpha n_seg + beta span + gamma D, times the fit's
  ``max_actual_over_fit``.
- **The split FCC reading** (Part A's ``leafref_07b.py --split-ratio 1.8``, its Route B FCC): Part A's formula (n_seg x
  1.1 x CPU per segment / 6 workers x 2; r1 0.097 s, r2 0.388 s) **plus** the FCC model's gamma D over the non-monotone
  segments only (splitting cannot make those monotone; the steep monotone ones are split and read in t), with the FCC
  model's margin on that term.
- The readings decided as Part A's section 2.4: split FCC r1 and r2 on every call; QAWO r1 and r2 (one_reading.py) where
  each projects at 30 minutes or less with its margin.
- The total: the readings one at a time, six worker processes each, and the same times 1.5 for load and scatter.

Usage: PYTHONPATH=<tree>:<rehearsal scripts/capture> python -B refs_projection_r31.py OUT_JSON
"""

import datetime
import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
Z = HERE.parents[1]
FITS = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/census/out/refs/refs_projection.json")
FITS_SHA = "0512daddfc3bb920cbb4d0d0515c8b735742c631bb69b76b56b2fd350af15a1b"
REFCAP = Z / "out" / "refcap.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- copied verbatim from 07's refs_projection_07.py ----
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


# ---- end of the verbatim copy ----


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    import ray
    import tools.filon_reference as F

    assert sha256(FITS) == FITS_SHA
    fits = json.loads(FITS.read_text())["fits"]
    caps = json.loads(REFCAP.read_text())["captures"]
    calls = []
    for c in caps:
        call = load_capture(c["pickle"], sha=c["sha256"])
        route = F.route_of(call)
        f = features(call)
        # the non-monotone segments' D alone (D_q counts exactly those)
        d_nonmono = f["D_q"]
        proj = {}
        for impl, res in (("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2)):
            k = f"{impl}_r{res}"
            al, be, ga = fits[k]["alpha_beta_gamma"]
            w = al * f["n_seg"] + be * f["span_q" if impl == "qawo" else "span_f"] + ga * f["D_q" if impl == "qawo" else "D_f"]
            proj[k] = {"fit_s": w, "with_margin_s": w * max(1.0, fits[k]["max_actual_over_fit"])}
        for r, cps in ((1, 0.097), (2, 0.388)):
            ga = fits[f"fcc_r{r}"]["alpha_beta_gamma"][2]
            mg = max(1.0, fits[f"fcc_r{r}"]["max_actual_over_fit"])
            base = f["n_seg"] * 1.1 * cps / 6.0
            proj[f"fccsplit_r{r}"] = {"fit_s": base + ga * d_nonmono, "with_margin_s": 2.0 * base + mg * ga * d_nonmono}
        qawo_ok = all(proj[f"qawo_r{r}"]["with_margin_s"] <= 1800.0 for r in (1, 2))
        readings = ([("qawo", 1), ("qawo", 2)] if qawo_ok else []) + [("fccsplit", 1), ("fccsplit", 2)]
        w = sum(proj[f"{i}_r{r}"]["with_margin_s"] for i, r in readings)
        interest7 = c["record"] != "WKB_Levin.subintervals[1].groups[0]" or c["label"] != "G+q+r"
        calls.append({"key": c["key"], "record": c["record"], "label": c["label"], "pickle": c["pickle"], "sha256": c["sha256"],
                      "in_the_seven": interest7, "route": route, "features": f, "projection": proj, "qawo": qawo_ok,
                      "readings": readings, "total_with_margin_s": w,
                      "e30_e31": (c["R30"]["abserr"], c["R31"]["abserr"])})
        print(f"{c['key']} {c['label']} (seven: {interest7}) route {route}: n_seg {f['n_seg']}, non-monotone {f['n_nonmono']} "
              f"(D {d_nonmono:.3g} rad), steep {f['n_steep']} (D_f {f['D_f']:.3g}); "
              + " ".join(f"{k} {v['with_margin_s']:.0f}" for k, v in proj.items())
              + f" | {'QAWO+split FCC' if qawo_ok else 'split FCC only'}: {w:.0f} s")
    tot7 = sum(x["total_with_margin_s"] for x in calls if x["in_the_seven"])
    tot8 = sum(x["total_with_margin_s"] for x in calls)
    doc = {"header": {"script": __file__, "script_sha256": sha256(__file__), "host": socket.gethostname(),
                      "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "fits": str(FITS), "fits_sha256": FITS_SHA, "refcap": str(REFCAP), "refcap_sha256": sha256(REFCAP),
                      "tool": F.__file__, "tool_sha256": sha256(F.__file__)},
           "calls": calls, "total_seven_with_margin_s": tot7, "total_eight_with_margin_s": tot8,
           "total_seven_x1_5_s": 1.5 * tot7, "total_eight_x1_5_s": 1.5 * tot8, "ray_initialized": ray.is_initialized()}
    out_p.write_text(json.dumps(doc, indent=1))
    print(f"the seven: {tot7:.0f} s with margin ({tot7 / 3600:.2f} h), x1.5 {1.5 * tot7 / 3600:.2f} h; all eight {tot8:.0f} s "
          f"({tot8 / 3600:.2f} h), x1.5 {1.5 * tot8 / 3600:.2f} h; ray initialised {doc['ray_initialized']}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
