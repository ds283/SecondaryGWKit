"""
levin-phase-floor prompt 05a section 6: T1 scored by the true step, as the user decided after the
run (relayed 2026-10-26 ~13:23Z): "T1 is scored on windows whose true step |dtheta/dz|(b-a)/64 is
below pi/2, narrowing further where it isn't ... The literal unwrapped-step reading is reported
beside it, unscored." New in 05a, after the run. Reads ``state/probe_windows.json`` only; imports
nothing from either tree and runs no probe.

analyse.py's (and rescore_t1.py's) "true_step" reading chose the first level at which BOTH the
true step and the literal unwrapped step were below pi/2 in both trees. The user's rule names the
true step alone. This script:
1. finds, per window, the first level at which the true step is below pi/2 in both c and t
   (``true_step_z_est``, the same double in both trees since theta_deriv is unchanged), with both
   trees' 65 z samples distinct; none: unresolved (section 9);
2. checks that this level equals the level analyse.py used for its "true_step" reading; where it
   does, the samples saved at that level are the window's, and sigma under c and t is re-computed
   from them; where it does not, it says so (that window would need its samples from the run's
   pickles, or a re-probe);
3. applies section 9's rule, otherwise unchanged, and prints the verdict and its numbers.

Usage: python -I -B t1_true_step_only.py PROBE_JSON
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
    with open(__file__, "rb") as f:
        print(f"# t1_true_step_only.py sha256 {hashlib.sha256(f.read()).hexdigest()}")
    print(f"# input {path} sha256 {digest}")
    P = json.load(open(path))
    tally = collections.Counter()
    differ, need_reprobe = [], []
    c_r, t_r, t_b, lit_steps, levels = [], [], [], [], collections.Counter()
    for w in P["windows"]:
        lc, lt = w["ladder"]["c"], w["ladder"]["t"]
        lev = None
        for a, b in zip(lc, lt):
            if not (a["distinct"] and b["distinct"]):
                break
            if a["true_step_z_est"] < math.pi / 2 and b["true_step_z_est"] < math.pi / 2:
                lev = a["n"]
                break
        rec = w["true_step"]
        used = None if rec is None else rec["n"]
        tag = (w["row"], w["subinterval"], w["factor"])
        if lev != used:
            differ.append((tag, lev, used))
        scale = EPS * w["abs_theta"]
        if lev is None:
            tally["unresolved" + ("" if scale >= 1e-11 else " (below 1e-11)")] += 1
            if scale >= 1e-11:
                need_reprobe.append(tag)
            continue
        if scale < 1e-11:
            tally["below 1e-11, not scored"] += 1
            continue
        if lev != used:
            need_reprobe.append(tag)
            continue
        s = rec["samples"]
        x = H(s["z"])
        sc, stc = fit(x, H(s["theta_z_c"]))
        st, stt = fit(x, H(s["theta_z_t"]))
        pz = float(np.max(np.abs(H(s["p_z_t"]))))
        bound = max(1e-3 * scale, EPS * (w["k"] * pz + 2 * math.pi), 1e-13)
        ok = sc >= 0.1 * scale and st <= bound
        tally["true" if ok else "false"] += 1
        c_r.append(sc / scale)
        t_r.append(st / scale)
        t_b.append(st / bound)
        lit_steps.append(max(stc, stt))
        levels[lev] += 1
    verdict = "fails" if tally["false"] else ("undecided" if tally["unresolved"] else "holds")
    print(f"levels where the true-step-alone choice differs from analyse.py's true_step level: "
          f"{differ if differ else 'none'}")
    print(f"scored windows needing a re-probe (no kept level resolves them, or their level's samples "
          f"are not saved): {need_reprobe if need_reprobe else 'none'}")
    print(f"T1 [true step alone, SCORED per the user's decision]: {dict(tally)} -> {verdict.upper()}")
    if c_r:
        print(f"   sigma_c/eps|theta| in [{min(c_r):.4g}, {max(c_r):.4g}]; sigma_t/eps|theta| in "
              f"[{min(t_r):.4g}, {max(t_r):.4g}]; largest sigma_t/bound {max(t_b):.4g}; largest unwrapped "
              f"step at the level used {max(lit_steps):.4g} rad; levels used {dict(sorted(levels.items()))}")


if __name__ == "__main__":
    main(sys.argv[1])
