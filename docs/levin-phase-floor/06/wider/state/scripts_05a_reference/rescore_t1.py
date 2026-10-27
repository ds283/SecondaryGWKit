"""
levin-phase-floor prompt 05a section 6: re-score T1 (as amended by section 9) from
``state/probe_windows.json`` alone. New in 05a. It imports nothing from either tree and reads no
checkpoint, so that T1 can be re-scored from the saved windows (orchestrator note, review check 9).

For each window it re-derives the level used from both trees' ladders (both readings: the literal
"largest unwrapped z-step < pi/2 in both trees", and the unscored "also the true step
|d theta/dz| (b - a)/64 < pi/2"), re-computes sigma_theta under c and t from the saved samples
(``numpy.unwrap``, degree-4 fit in t in [-1, 1], RMS; test_t_phase_reanchor._sigma_theta),
max |p_z| under t from the saved p_z, and applies section 9's rule:
  sigma_c >= 0.1 eps|theta_T|  and  sigma_t <= max(1e-3 eps|theta_T|, eps (k max|p_z| + 2 pi), 1e-13)
in every resolved window with eps|theta_T| >= 1e-11 rad.

Usage: python -I -B rescore_t1.py PROBE_JSON
"""

import collections
import hashlib
import json
import math
import sys

import numpy as np

EPS = float(np.finfo(np.float64).eps)


def fit(x, th):
    a, b = x[0], x[-1]
    u = np.unwrap(th)
    t = (2.0 * x - (a + b)) / (b - a)
    coef = np.polynomial.polynomial.polyfit(t, u, 4)
    res = u - np.polynomial.polynomial.polyval(t, coef)
    return float(np.sqrt(np.mean(res * res))), float(np.max(np.abs(np.diff(u))))


def H(v):
    return np.array([float.fromhex(s) for s in v])


def main(path):
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(path) as f:
        P = json.load(f)
    me = __file__
    with open(me, "rb") as f:
        print(f"# rescore_t1.py sha256 {hashlib.sha256(f.read()).hexdigest()}")
    print(f"# input {path} sha256 {digest}; {len(P['windows'])} windows")
    for reading in ("literal", "true_step"):
        tally = collections.Counter()
        bad = []
        for w in P["windows"]:
            lc, lt = w["ladder"]["c"], w["ladder"]["t"]
            lev = None
            for a, b in zip(lc, lt):
                if not (a["distinct"] and b["distinct"]):
                    break
                if a["step_z"] < math.pi / 2 and b["step_z"] < math.pi / 2 and (
                    reading == "literal"
                    or (a["true_step_z_est"] < math.pi / 2 and b["true_step_z_est"] < math.pi / 2)
                ):
                    lev = a["n"]
                    break
            rec = w[reading]
            if (lev is None) != (rec is None) or (lev is not None and lev != rec["n"]):
                bad.append(f"{w['row']} {w['subinterval']} {w['factor']}: level {lev} vs {rec and rec['n']}")
            scale = EPS * w["abs_theta"]
            if scale < 1e-11:
                tally["below 1e-11, not scored"] += 1
                continue
            if lev is None:
                tally["unresolved"] += 1
                continue
            s = rec["samples"]
            x = H(s["z"])
            sc, stc = fit(x, H(s["theta_z_c"]))
            st, stt = fit(x, H(s["theta_z_t"]))
            pz = float(np.max(np.abs(H(s["p_z_t"]))))
            if (sc, st, stc, stt, pz) != (rec["sigma_c"], rec["sigma_t"], rec["step_c"], rec["step_t"], rec["max_abs_p_z_t"]):
                bad.append(f"{w['row']} {w['subinterval']} {w['factor']}: recomputed numbers differ")
            bound = max(1e-3 * scale, EPS * (w["k"] * pz + 2 * math.pi), 1e-13)
            ok = sc >= 0.1 * scale and st <= bound
            tally["true" if ok else "false"] += 1
        verdict = "fails" if tally["false"] else ("undecided" if tally["unresolved"] else "holds")
        print(f"T1 [{reading}]: {dict(tally)} -> {verdict.upper()}; disagreements with analyse.py: "
              f"{bad if bad else 'none'}")


if __name__ == "__main__":
    main(sys.argv[1])
