"""levin-phase-floor prompt 07c Part A extension (R2-28), questions 2 and 3: named spans of R2-28 re-solved at the tree's
module (07b's code, 8956c7b6..., imported unchanged from the worktree), at the next depths and by second rules. New in
the extension, after 07b Part A's leafprobe_07b.py (a record), whose solve and break-point expressions it keeps.

For each span [a, b] named in the probe spec (JSON list of {"name", "lo", "hi"} in hex), with the call's captured f and
theta and the call's (rtol, chebyshev_order N):
- **the driver's solve**: ``_adaptive_levin_subregion((a, b), f, _Basis_SinCos(theta), ...)`` (the gate decides Levin or
  Clenshaw-Curtis, as the driver's ``_solve`` does for a leaf not forced direct) on P = [a, b], its halves, quarters and
  eighths (c = (b + a)/2, quarters (c + a)/2 and (b + c)/2, each quarter's midpoint the same way); V_P, the halves' sum,
  the quarters' sum Q and the eighths' sum, and each piece's value, route (``is_direct``), phase variation V, floor
  (``phase_err``) and truncation;
- **a second rule on the same pieces**: (i) ``force_direct=True`` (the driver's nested Clenshaw-Curtis pair, CC_{2N-1});
  (ii) the Levin solve at order N + 8 and N - 8 (the gate is unchanged by the order, so a piece the gate sends to
  Clenshaw-Curtis is Clenshaw-Curtis at every order);
- **the phase rate**: |theta'| at 513 Chebyshev points of the span (the captured ``theta_deriv``): min, max and their
  ratio.
Nothing is compared here; r2_28_table_07c.py sets each value against the leaf references (out/r2-28/leafref.json).

Writes OUT_JSON (refuses to overwrite). Usage: PYTHONPATH=<tree>:<scripts> python -B r2_28_probe_07c.py CAPTURE_SPEC_JSON
PROBE_SPEC_JSON OUT_JSON   (CAPTURE_SPEC_JSON: state/r2-28/spec.json, whose "capture" is read)
"""

import datetime
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f")
TREE_LQ_SHA = "8956c7b6a8ec285f8898353eed65f4039a02b42edc6927565be3896cda52def1"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def H(x):
    return None if x is None else float(x).hex()


def subdivide(a, b):
    c = (b + a) / 2.0
    cL, cR = (c + a) / 2.0, (b + c) / 2.0
    sp = {"P": (a, b), "L": (a, c), "R": (c, b), "LL": (a, cL), "LR": (cL, c), "RL": (c, cR), "RR": (cR, b)}
    for q in ("LL", "LR", "RL", "RR"):
        lo, hi = sp[q]
        m = (hi + lo) / 2.0
        sp[q + "L"], sp[q + "R"] = (lo, m), (m, hi)
    return sp


def main():
    cap_spec, probe_spec, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if out.exists():
        raise SystemExit(f"{out} exists")
    if os.environ.get("PYTHONPATH", "").split(":")[0] != str(TREE):
        raise SystemExit("PYTHONPATH must start with the tree")
    import AdaptiveLevin.levin_quadrature as LQ

    if sha256(LQ.__file__) != TREE_LQ_SHA:
        raise SystemExit("the tree's module has another SHA-256")
    import ray.cloudpickle as cp

    c = json.loads(cap_spec.read_text())["capture"]
    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture blob has another SHA-256")
    rec = cp.loads(blob)
    kw = dict(rec["kw"])
    f_, theta = rec["f"], rec["theta"]
    N = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    rtol = kw["rtol"]
    A = LQ._Basis_SinCos(theta)

    def solve(span, force=False, order=N):
        return LQ._adaptive_levin_subregion(span, f_, A, id_label="r2-28-probe-07c", chebyshev_order=order, rtol=rtol,
                                            notify_label=rec["notify_label"], vectorize_cache={}, force_direct=force)

    def piece(d):
        return {"value": H(d["value"]), "is_direct": bool(d.get("is_direct", False)), "phase_span": H(d.get("phase_span")),
                "phase_err": H(d.get("phase_err")), "abserr_truncation": H(d.get("abserr_truncation")),
                "abserr_direct": H(d.get("abserr_direct")), "chebyshev_order": d["metadata"].get("chebyshev_order")}

    def sums(vals):
        return {"V_P": H(vals["P"]), "S1": H(vals["L"] + vals["R"]),
                "Q": H((vals["LL"] + vals["LR"]) + (vals["RL"] + vals["RR"])),
                "S3": H(sum((vals[q + "L"] + vals[q + "R"] for q in ("LL", "LR", "RL", "RR")), 0.0))}

    names = json.loads(probe_spec.read_text())
    doc = {"header": {"title": "levin-phase-floor 07c Part A extension: R2-28 span probes",
                      "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "module": LQ.__file__, "module_sha256": TREE_LQ_SHA, "capture": c,
                      "probe_spec": str(probe_spec), "probe_spec_sha256": sha256(probe_spec), "N": N, "rtol": H(rtol)},
           "spans": []}
    t0 = time.perf_counter()
    for e in names:
        a, b = float.fromhex(e["lo"]), float.fromhex(e["hi"])
        sp = subdivide(a, b)
        rules = {}
        for rname, kwr in (("driver", {}), ("cc", {"force": True}), ("levin_Np8", {"order": N + 8}), ("levin_Nm8", {"order": N - 8})):
            ds = {k: solve(s, **kwr) for k, s in sp.items()}
            rules[rname] = {"pieces": {k: piece(d) for k, d in ds.items()}, **sums({k: d["value"] for k, d in ds.items()})}
        xs = (a + b) / 2.0 + (b - a) / 2.0 * np.cos(np.pi * np.arange(513) / 512.0)
        tp = np.abs(np.asarray([theta["theta_deriv"](x) for x in xs], dtype=float))
        doc["spans"].append({**e, "spans": {k: [H(s[0]), H(s[1])] for k, s in sp.items()}, "rules": rules,
                             "theta_prime_abs": {"min": H(tp.min()), "max": H(tp.max()), "ratio": H(tp.max() / tp.min())}})
        print(f"{e['name']} [{a:.6g}, {b:.6g}]: driver V_P {float.fromhex(rules['driver']['V_P']):.17g} "
              f"Q {float.fromhex(rules['driver']['Q']):.17g}; |theta'| {tp.min():.4g}..{tp.max():.4g}", flush=True)
    doc["wall_s"] = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print("wrote", out, sha256(out))


if __name__ == "__main__":
    main()
