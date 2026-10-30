"""Y1 references, step 4 (sizing): replay each pickled call in its own tree and measure its phase.

Run with PYTHONPATH=<tree>:<scripts>. For each pickle under PICKLE_DIR named <row>-<tree>-<idx>.pkl
of this process's tree (and of the rows given):
1. load it (ray.cloudpickle), re-run adaptive_levin_sincos(x_span, f, theta=theta, **kw), and check
   that value and abserr equal the captured call's, and (1 + z_response) * each equals the store's
   group metadata, bit for bit (float.hex);
2. measure the phase on the span: theta (raw) and theta_mod_2pi at both ends, the oscillation count
   (theta(b) - theta(a)) / 2 pi, theta' (the dict's theta_deriv) on a uniform grid of NGRID points in
   x: min |theta'| and where, max |theta'|, sign changes, and the integral of |theta'| (trapezoid);
3. time one evaluation of the integrand's ingredients (f_sin, f_cos, theta_mod_2pi) per node.

Usage: PYTHONPATH=<tree>:<scripts> python -B probe.py PICKLE_DIR OUT_JSON [--rows Y01,..] [--ngrid N]
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from yprov import header, which_tree  # noqa: E402


def hx(x):
    return None if x is None else float(x).hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle_dir")
    ap.add_argument("out_json")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--ngrid", type=int, default=20001)
    a = ap.parse_args()
    tree = which_tree()
    if tree is None:
        raise SystemExit("PYTHONPATH must name one of the two trees")
    files = sorted(Path(a.pickle_dir).glob(f"*-{tree}-*.pkl"))
    if a.rows:
        want = set(a.rows.split(","))
        files = [f for f in files if f.name.split("-")[0] in want]
    hdr = header(Path(__file__).resolve(), f"Y1 references: replay and phase probe in tree {tree}", files)
    for l in hdr:
        print(l, flush=True)
    import numpy as np
    import ray.cloudpickle as cloudpickle
    import AdaptiveLevin.levin_quadrature as LQ

    print(f"# AdaptiveLevin.levin_quadrature {LQ.__file__}", flush=True)
    out = {"header": hdr, "tree": tree, "calls": []}
    for fp in files:
        blob = fp.read_bytes()
        rec = cloudpickle.loads(blob)
        d = {"pickle": str(fp), "pickle_sha256": hashlib.sha256(blob).hexdigest(), "row": rec["row"],
             "index": rec["index"], "label": rec["label"]}
        x0, x1 = rec["x_span"]
        theta = rec["theta"]
        f_sin, f_cos = rec["f"]
        t0 = time.perf_counter()
        rr = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], theta=theta, **rec["kw"])
        d["replay_seconds"] = time.perf_counter() - t0
        scale = float.fromhex(rec["scale"])
        d["replay_value"] = hx(rr["value"])
        d["replay_abserr"] = hx(rr["abserr"])
        d["replay_equals_capture"] = (hx(rr["value"]), hx(rr["abserr"])) == (rec["value"], rec["abserr"])
        d["replay_equals_store"] = (hx(scale * rr["value"]), hx(scale * rr["abserr"])) == (rec["store_scaled_value"], rec["store_scaled_abserr"])
        d["replay_regions_depth"] = [int(rr["num_regions"]), int(rr["max_depth"])] == [rec["num_regions"], rec["max_depth"]]
        th, thm, thd = theta["theta"], theta["theta_mod_2pi"], theta["theta_deriv"]
        d["x_span"] = [hx(x0), hx(x1)]
        d["theta_a"], d["theta_b"] = hx(th(x0)), hx(th(x1))
        d["theta_mod_a"], d["theta_mod_b"] = hx(thm(x0)), hx(thm(x1))
        osc = (th(x1) - th(x0)) / (2 * math.pi)
        d["oscillations"] = osc
        xs = np.linspace(x0, x1, a.ngrid)
        t0 = time.perf_counter()
        tp = np.array([thd(float(x)) for x in xs])
        d["seconds_per_theta_deriv"] = (time.perf_counter() - t0) / a.ngrid
        atp = np.abs(tp)
        i = int(np.argmin(atp))
        d["min_abs_theta_deriv"] = float(atp[i])
        d["x_at_min"] = float(xs[i])
        d["z_at_min"] = math.expm1(float(xs[i]))
        d["max_abs_theta_deriv"] = float(atp.max())
        d["mean_abs_theta_deriv"] = float(atp.mean())
        d["sign_changes"] = int(np.sum(np.signbit(tp[1:]) != np.signbit(tp[:-1])))
        integ = float(np.trapezoid(atp, xs))
        d["int_abs_theta_deriv_over_2pi"] = integ / (2 * math.pi)
        # per-node cost of the integrand's ingredients
        ns = 2000
        xt = np.linspace(x0, x1, ns + 2)[1:-1] * (1 + 1e-13)
        xt = np.clip(xt, x0, x1)
        t0 = time.perf_counter()
        for x in xt:
            x = float(x)
            f_sin(x), f_cos(x), thm(x)
        d["seconds_per_node"] = (time.perf_counter() - t0) / ns
        d["regions"] = [[hx(r.start), hx(r.end), int(r.depth), str(r.type)] for r in rr["regions"]]
        out["calls"].append(d)
        print(f"{d['row']} {d['index']} {d['label']:<6} replay=capture {d['replay_equals_capture']} =store {d['replay_equals_store']} "
              f"reg/depth {d['replay_regions_depth']}  span [{x0:.5f},{x1:.5f}]  osc {osc:.4g}  int|th'|/2pi {d['int_abs_theta_deriv_over_2pi']:.4g}  "
              f"min|th'| {d['min_abs_theta_deriv']:.4g} at z={d['z_at_min']:.5g} (max {d['max_abs_theta_deriv']:.4g}, signchg {d['sign_changes']})  "
              f"node {d['seconds_per_node'] * 1e6:.1f} us  replay {d['replay_seconds']:.2f} s", flush=True)
    Path(a.out_json).write_text(json.dumps(out, indent=1))
    ok = all(c["replay_equals_capture"] and c["replay_equals_store"] and c["replay_regions_depth"] for c in out["calls"])
    print(f"calls {len(out['calls'])}; every replay reproduces capture and store bit for bit: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
