"""The four-block check (the user's word at R29's stop): one captured analytic block (2pt5 J4 or Y4) at one tree --
its replay, its samples, and (``--reference``, on R29's tree's capture) its reference, its phase variation V and the
C2 gate's span estimate on the root leaf. levin-phase-floor 07, rehearsal agent, R29. New, after qr_reference.py.

Run once per (block, tree), in a process whose PYTHONPATH is that tree, then r29/scripts/capture (the pickles name
harness._rebuild_lru_cache):

    python -B block_reference.py TREE PICKLE OUT_JSON [--reference]

1. The capture (written by the harness's worker in pass 2, checked by replay07_r29.py blockcheck against its store) is
   loaded with ray.cloudpickle; Ray is not initialised.
2. The replay: adaptive_levin_sincos(x_span, f, theta, **kw) from TREE; value, abserr (float.hex), num_regions and
   converged must equal the capture's record.
3. The samples: the integrand I(x) = f_sin(x) sin t(x) + f_cos(x) cos t(x), t = theta_mod_2pi (as the driver evaluates
   it), with theta and theta_deriv, at 1001 points uniform in x, by float.hex; compared across trees by block_score.py.
4. (--reference) The root: theta_deriv sampled at 40,001 points uniform on x_span; each sign change refined by brentq
   (xtol 4 eps |x|); there must be exactly one. V = int |theta'| dx by the trapezoidal rule on the driver's own 4,097-
   point scan (np.linspace(x_a, x_b, 4097)), as Rule A estimates it; the C2 gate's span estimate on the root leaf,
   mean|theta'(grid)| * |grid[0] - grid[-1]| with (grid, _) = levin_quadrature.chebyshev_matrices(x_span,
   kw["chebyshev_order"]), as build_Levin_data forms it. The reference: on each side of the root, composite 20-point
   Gauss-Legendre on 2^m equal x-panels, m = 6, 8, 10, 12, 14 (panel sums h/2 fsum(w I), the panels and the two sides
   by fsum); and QUADPACK's QAGS (scipy.integrate.quad, epsabs 0, epsrel 1.2e-14, limit 10000, points=[root]) as a
   second rule. The reference is GL at m = 14; its stated error is max(|GL14 - GL12|, |GL14 - QAGS|).
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    tree, pkl, out = sys.argv[1:4]
    do_ref = "--reference" in sys.argv[4:]
    assert sys.dont_write_bytecode
    import numpy as np
    import ray
    import ray.cloudpickle as cp
    import AdaptiveLevin.levin_quadrature as LQ

    head = subprocess.run(["git", "--no-optional-locks", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = subprocess.run(["git", "--no-optional-locks", "-C", tree, "status", "--short"], capture_output=True, text=True).stdout
    assert LQ.__file__.startswith(tree + os.sep), LQ.__file__
    res = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "pickle": pkl,
                      "pickle_sha256": sha256(pkl), "tree": tree, "tree_head": head, "tree_status_short": st,
                      "levin_quadrature_sha256": sha256(LQ.__file__), "host": os.uname().nodename,
                      "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    with open(pkl, "rb") as f:
        d = cp.load(f)
    res.update(unit=d["unit"], notify_label=d["notify_label"], x_span=[float(d["x_span"][0]).hex(), float(d["x_span"][1]).hex()],
               kw={k: (float(v).hex() if isinstance(v, float) else v) for k, v in d["kw"].items()},
               capture_record={k: d[k] for k in ("value", "abserr", "num_regions", "converged", "max_depth")})
    t0 = time.time()
    r = LQ.adaptive_levin_sincos(d["x_span"], d["f"], d["theta"], **d["kw"])
    res["replay"] = {"value": float(r["value"]).hex(), "abserr": float(r["abserr"]).hex(), "num_regions": int(r["num_regions"]),
                     "converged": bool(r["converged"]), "phase_limited": bool(r["phase_limited"]), "seconds": time.time() - t0,
                     "stationary_points": len(r["stationary_points"]) if "stationary_points" in r else None,
                     "num_guarded_regions": r.get("num_guarded_regions"),
                     "location_skipped_phases": [{"phase": s["phase"], "variation": float(s["variation"]).hex()}
                                                 for s in r["location_skipped_phases"]] if "location_skipped_phases" in r else None}
    res["replay_reproduces_capture"] = (res["replay"]["value"] == d["value"] and res["replay"]["abserr"] == d["abserr"]
                                        and res["replay"]["num_regions"] == d["num_regions"] and res["replay"]["converged"] == d["converged"])
    fs, fc = d["f"]
    th = d["theta"]
    tm = th.get("theta_mod_2pi") or th["theta"]

    def integrand(x):
        t = float(tm(x))
        return float(fs(x)) * math.sin(t) + float(fc(x)) * math.cos(t)

    a, b = float(d["x_span"][0]), float(d["x_span"][1])
    lo, hi = min(a, b), max(a, b)
    xs = [a + (b - a) * i / 1000 for i in range(1001)]
    res["samples"] = [[float(x).hex(), float(integrand(x)).hex(), float(th["theta"](x)).hex(), float(tm(x)).hex(),
                       float(th["theta_deriv"](x)).hex()] for x in xs]
    if do_ref:
        from scipy.integrate import quad
        from scipy.optimize import brentq

        td = th["theta_deriv"]
        grid40 = np.linspace(lo, hi, 40001)
        v40 = np.array([float(td(float(x))) for x in grid40])
        idx = [i for i in range(40000) if v40[i] != 0.0 and v40[i + 1] != 0.0 and np.sign(v40[i]) != np.sign(v40[i + 1])]
        zeros = [i for i in range(40001) if v40[i] == 0.0]
        roots = [brentq(lambda x: float(td(x)), float(grid40[i]), float(grid40[i + 1]), xtol=4 * 2.0 ** -52 * max(abs(grid40[i]), abs(grid40[i + 1])), rtol=8.9e-16)
                 for i in idx] + [float(grid40[i]) for i in zeros]
        res["roots"] = [x.hex() for x in roots]
        res["root_eta"] = [repr(math.exp(x)) for x in roots]
        assert len(roots) == 1, roots
        x0 = roots[0]
        scan = np.linspace(a, b, 4097)
        sv = np.abs(np.array([float(td(float(x))) for x in scan]))
        V = float(np.sum(0.5 * (sv[1:] + sv[:-1]) * np.abs(np.diff(scan))))
        grid, _ = LQ.chebyshev_matrices(d["x_span"], d["kw"]["chebyshev_order"])
        tpc = np.array([float(td(float(x))) for x in grid])
        span = float(np.mean(np.fabs(tpc)) * np.fabs(grid[0] - grid[-1]))
        res["V_trapezoid_4097"] = {"repr": repr(V), "hex": V.hex()}
        res["gate_span_root_leaf"] = {"repr": repr(span), "hex": span.hex(), "SIX_PI": repr(LQ.SIX_PI),
                                      "chebyshev_order": d["kw"]["chebyshev_order"], "grid_points": len(grid)}
        xg, wg = np.polynomial.legendre.leggauss(20)

        def gl(lo_, hi_, n):
            parts = []
            for p in range(n):
                p_lo = lo_ + (hi_ - lo_) * p / n
                p_hi = lo_ + (hi_ - lo_) * (p + 1) / n
                c, hw = 0.5 * (p_lo + p_hi), 0.5 * (p_hi - p_lo)
                parts.append(hw * math.fsum(float(wg[j]) * integrand(c + hw * float(xg[j])) for j in range(20)))
            return math.fsum(parts)

        sgn = 1.0 if b >= a else -1.0
        glv = {}
        t1 = time.time()
        for m in (6, 8, 10, 12, 14):
            n = 2 ** m
            v = sgn * math.fsum([gl(lo, x0, n), gl(x0, hi, n)])
            glv[m] = {"value": v.hex(), "repr": repr(v), "evaluations": 2 * 20 * n}
        t_gl = time.time() - t1
        q, qerr, info = quad(integrand, lo, hi, epsabs=0.0, epsrel=1.2e-14, limit=10000, points=[x0], full_output=1)[:3]
        q = sgn * q
        ref = float.fromhex(glv[14]["value"])
        err = max(abs(ref - float.fromhex(glv[12]["value"])), abs(ref - q))
        res["reference"] = {"method": "composite 20-point Gauss-Legendre on 2^m x-panels each side of the root (fsum), m = 6..14; QAGS (epsrel 1.2e-14, points=[root])",
                            "GL": glv, "QAGS": {"value": float(q).hex(), "repr": repr(float(q)), "abserr": repr(float(qerr)),
                                                "neval": int(info["neval"]), "last": int(info["last"])},
                            "reference": ref.hex(), "reference_repr": repr(ref), "stated_error": repr(err), "stated_error_hex": float(err).hex(),
                            "differences": {f"GL{m}-GL14": repr(float.fromhex(glv[m]["value"]) - ref) for m in (6, 8, 10, 12)} | {"QAGS-GL14": repr(q - ref)},
                            "seconds_GL": t_gl}
    res["ray_initialized"] = ray.is_initialized()
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
