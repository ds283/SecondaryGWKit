"""References for three q-r group calls with q = r (theta' identically zero), and the replay of their captures.

levin-phase-floor 07, rehearsal agent, the user's addendum at planned stop 1 ("add a reference to check 2 or
3 of the q-r calls, to settle whether R28 is honest"). New.

Run once per (call, tree), in a process whose PYTHONPATH is that tree, then the capture harness's
directory (rehearsal/scripts/capture, for harness._rebuild_lru_cache, which the pickles name):

    python -B qr_reference.py TREE PICKLE OUT_JSON [--reference]

1. **The capture.** PICKLE (written by the capture harness's worker, checked by replay07.py capcheck against
   the store) is loaded with ray.cloudpickle (Ray is not initialised). It holds the call's x_span, f, theta
   and kw, and the worker's record of the call (value, abserr, num_regions, converged).
2. **The replay.** adaptive_levin_sincos(x_span, f, theta, **kw) from TREE; its value and abserr (float.hex),
   num_regions and converged must equal the capture's record (so the pickle is the stored call's integrand).
3. **The samples.** The integrand I(x) = f_sin(x) sin(t(x)) + f_cos(x) cos(t(x)), t = theta_mod_2pi (as the
   driver evaluates sin and cos when theta_mod_2pi is supplied), at 1001 points x_i = a + (b - a) i / 1000,
   recorded by float.hex, with theta, theta_mod_2pi and theta_deriv there. The two trees' captures of one
   call are compared on these by a separate step (bit for bit).
4. **The reference** (``--reference``; run on R28's tree's capture). Since theta' is identically zero the
   integrand does not oscillate, so a direct quadrature of the captured integrand, evaluated in double by its
   own callables, is used:
   - composite 20-point Gauss-Legendre on 2^m equal panels in x, m = 6, 8, 10, 12, 14; each panel's sum
     h/2 * fsum(w_j I(x_j)), the panels summed by math.fsum;
   - QUADPACK's QAGS (scipy.integrate.quad, epsabs 0, epsrel 1.2e-14 -- scipy's least above 50 eps -- limit 10000), an independent adaptive
     rule.
   The reference is GL at m = 14. Its stated error is max(|GL14 - GL12|, |GL14 - QAGS|), and beside it the
   rounding scale eps * sum |h/2 w_j I(x_j)| at m = 14 and QAGS's own estimate.

Writes OUT_JSON with a provenance header (script and pickle SHA-256, the tree's HEAD and status).
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


def tree_state(t):
    head = subprocess.run(["git", "--no-optional-locks", "-C", t, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = subprocess.run(["git", "--no-optional-locks", "-C", t, "status", "--short"], capture_output=True, text=True).stdout
    return head, st


def main():
    tree, pkl, out = sys.argv[1:4]
    do_ref = "--reference" in sys.argv[4:]
    assert sys.dont_write_bytecode
    import numpy as np
    import ray
    import ray.cloudpickle as cp
    import AdaptiveLevin.levin_quadrature as LQ

    head, st = tree_state(tree)
    hdr = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "pickle": pkl,
           "pickle_sha256": sha256(pkl), "tree": tree, "tree_head": head, "tree_status_short": st,
           "levin_quadrature": LQ.__file__, "levin_quadrature_sha256": sha256(LQ.__file__),
           "PYTHONPATH": os.environ.get("PYTHONPATH"), "host": os.uname().nodename,
           "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    assert LQ.__file__.startswith(tree + os.sep), LQ.__file__
    with open(pkl, "rb") as f:
        d = cp.load(f)
    res = {"header": hdr, "unit": d["unit"], "notify_label": d["notify_label"],
           "x_span": [float(d["x_span"][0]).hex(), float(d["x_span"][1]).hex()],
           "kw": {k: (float(v).hex() if isinstance(v, float) else v) for k, v in d["kw"].items()},
           "capture_record": {k: d[k] for k in ("value", "abserr", "num_regions", "converged", "max_depth")}}
    # 2. the replay
    t0 = time.time()
    r = LQ.adaptive_levin_sincos(d["x_span"], d["f"], d["theta"], **d["kw"])
    res["replay"] = {"value": float(r["value"]).hex(), "abserr": float(r["abserr"]).hex(), "num_regions": int(r["num_regions"]),
                     "converged": bool(r["converged"]), "phase_limited": bool(r["phase_limited"]),
                     "num_stationary_points": len(r.get("stationary_points") or []) if "stationary_points" in r else None,
                     "num_window_pieces": r.get("num_window_pieces"), "seconds": time.time() - t0}
    res["replay_reproduces_capture"] = (res["replay"]["value"] == d["value"] and res["replay"]["abserr"] == d["abserr"]
                                        and res["replay"]["num_regions"] == d["num_regions"]
                                        and res["replay"]["converged"] == d["converged"])
    # 3. the samples
    fs, fc = d["f"]
    th = d["theta"]
    tm = th.get("theta_mod_2pi") or th["theta"]

    def integrand(x):
        t = float(tm(x))
        return float(fs(x)) * math.sin(t) + float(fc(x)) * math.cos(t)

    a, b = float(d["x_span"][0]), float(d["x_span"][1])
    xs = [a + (b - a) * i / 1000 for i in range(1001)]
    res["samples"] = [[float(x).hex(), float(integrand(x)).hex(), float(th["theta"](x)).hex(), float(tm(x)).hex(),
                       float(th["theta_deriv"](x)).hex()] for x in xs]
    res["samples_theta_deriv_all_zero"] = all(float.fromhex(s[4]) == 0.0 for s in res["samples"])
    res["samples_theta_all_zero"] = all(float.fromhex(s[2]) == 0.0 for s in res["samples"])
    res["samples_integrand_sign"] = sorted({(float.fromhex(s[1]) > 0) - (float.fromhex(s[1]) < 0) for s in res["samples"]})
    # 4. the reference
    if do_ref:
        from scipy.integrate import quad

        xg, wg = np.polynomial.legendre.leggauss(20)
        gl = {}
        t1 = time.time()
        for m in (6, 8, 10, 12, 14):
            n = 2 ** m
            h = (b - a) / n
            parts, absparts = [], []
            for p in range(n):
                lo = a + (b - a) * p / n
                hi = a + (b - a) * (p + 1) / n
                c, hw = 0.5 * (lo + hi), 0.5 * (hi - lo)
                terms = [float(wg[j]) * integrand(c + hw * float(xg[j])) for j in range(20)]
                parts.append(hw * math.fsum(terms))
                absparts.append(hw * math.fsum(abs(t) for t in terms))
            v = math.fsum(parts)
            gl[m] = {"value": v.hex(), "repr": repr(v), "abs_sum": repr(math.fsum(absparts)), "evaluations": 20 * n}
        t_gl = time.time() - t1
        t2 = time.time()
        q, qerr, info = quad(integrand, a, b, epsabs=0.0, epsrel=1.2e-14, limit=10000, full_output=1)[:3]
        t_q = time.time() - t2
        ref = float.fromhex(gl[14]["value"])
        err = max(abs(ref - float.fromhex(gl[12]["value"])), abs(ref - q))
        res["reference"] = {
            "method": "composite 20-point Gauss-Legendre on 2^m equal x-panels (fsum), m = 6..14; and QAGS (epsrel 1.2e-14)",
            "GL": gl, "QAGS": {"value": float(q).hex(), "repr": repr(float(q)), "abserr": repr(float(qerr)),
                               "neval": int(info["neval"]), "last": int(info["last"])},
            "reference": ref.hex(), "reference_repr": repr(ref),
            "stated_error": repr(err), "stated_error_hex": float(err).hex(),
            "rounding_scale_eps_abs_sum_m14": repr(2.0 ** -52 * float(gl[14]["abs_sum"])),
            "differences": {f"GL{m}-GL14": repr(float.fromhex(gl[m]["value"]) - ref) for m in (6, 8, 10, 12)}
                           | {"QAGS-GL14": repr(q - ref)},
            "seconds_GL": t_gl, "seconds_QAGS": t_q,
        }
    res["ray_initialized"] = ray.is_initialized()
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
