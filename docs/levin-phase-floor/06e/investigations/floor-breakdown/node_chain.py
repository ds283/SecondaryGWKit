"""Where the CC node term's pessimism comes from, factor by factor, on the driver's CC leaves.

For each CC leaf of driver_run.json, with x^_k the computed fine nodes, x_k the exact ones,
dx_k = x^_k - x_k (exact, mpmath), w_k the weights, g = f_1 sin theta + f_2 cos theta:
  A = C3, the code's term:  delta * sum_k |w_k| (theta'_max sum_i |f_ik| + sum_i |f'_ik|)
  B = A with delta replaced by each node's actual |dx_k|
  C = B with theta'_max replaced by |theta'(x_k)|
  D = sum_k |w_k| |dx_k| |g'(x_k)|               (the true |g'|, not |theta'| sum|f| + sum|f'|)
  E = |sum_k w_k dx_k g'(x_k)|                     (first-order prediction of the node error)
  R = the measured node part (mp at the float nodes minus mp at the exact nodes)
A/B, B/C, C/D, D/E are the factors; E against R checks the first-order model.
Writes node_chain.json.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

lq = C.lq


def chain(case, a, b, measured):
    N = 2 * C.ORDER - 1
    f, B, cache = C.context(case)
    grid, Dm = lq.chebyshev_matrices((a, b), N)
    coarse, _ = lq.chebyshev_matrices((a, b), C.ORDER)
    tpmax = float(np.max(np.fabs(np.asarray(B._theta_deriv(coarse), dtype=float))))
    w = lq._cc_weights((a, b), N)
    delta = lq._cc_node_error_bound((a, b))
    with mp.workdps(C.MP_DPS):
        th, thp, f1, f2 = C._mp_funcs(case)
        A_, B_ = mp.mpf(a), mp.mpf(b)
        mid, half = (A_ + B_) / 2, (B_ - A_) / 2
        xe = [mid + half * mp.cos(mp.pi * k / (N - 1)) for k in range(N)]
        dx = [float(mp.mpf(grid[k]) - xe[k]) for k in range(N)]
        gp, tp, fs, fps = [], [], [], []
        for x in xe:
            t = th(x)
            F1, F2 = f1(x), f2(x)
            F1p, F2p = -F1, -3 * mp.sin(3 * x)
            gp.append(float(F1p * mp.sin(t) + F2p * mp.cos(t) + thp(x) * (F1 * mp.cos(t) - F2 * mp.sin(t))))
            tp.append(float(abs(thp(x))))
            fs.append(float(abs(F1) + abs(F2)))
            fps.append(float(abs(F1p) + abs(F2p)))
    aw = np.fabs(w)
    fs, fps, tp, gp, dx = map(np.array, (fs, fps, tp, gp, dx))
    A = delta * float(np.dot(aw, tpmax * fs + fps))
    Bv = float(np.dot(aw, np.fabs(dx) * (tpmax * fs + fps)))
    Cv = float(np.dot(aw, np.fabs(dx) * (tp * fs + fps)))
    D = float(np.dot(aw, np.fabs(dx) * np.fabs(gp)))
    E_signed = float(np.dot(w, dx * gp))
    return {
        "a": a, "b": b, "A_C3": A, "B": Bv, "C": Cv, "D": D, "E_signed": E_signed, "R_measured": measured,
        "A/B": A / Bv if Bv else None, "B/C": Bv / Cv if Cv else None, "C/D": Cv / D if D else None,
        "D/|E|": D / abs(E_signed) if E_signed else None,
        "E/R": E_signed / measured if measured else None,
        "delta_u": delta * 2**53, "max_dx_u": float(np.max(np.fabs(dx))) * 2**53,
    }


def main():
    d = json.loads((HERE / "driver_run.json").read_text())
    out = {"header": C.header(__file__), "cases": {}}
    for case in C.CASES:
        rows = [
            chain(case, r["a"], r["b"], r["roundoff_node_part"])
            for r in d["cases"][case["name"]]["leaves"]
            if r["kind"] == "CC"
        ]
        tot = {k: math.fsum(r[k] for r in rows) for k in ("A_C3", "B", "C", "D")}
        tot["sum_abs_E"] = math.fsum(abs(r["E_signed"]) for r in rows)
        tot["abs_sum_E"] = abs(math.fsum(r["E_signed"] for r in rows))
        tot["sum_abs_R"] = math.fsum(abs(r["R_measured"]) for r in rows)
        tot["abs_sum_R"] = abs(math.fsum(r["R_measured"] for r in rows))
        med = {k: float(np.median([r[k] for r in rows if r[k] is not None])) for k in ("A/B", "B/C", "C/D", "D/|E|", "E/R", "max_dx_u", "delta_u")}
        out["cases"][case["name"]] = {"totals": tot, "median_factors": med, "rows": rows}
        print(case["name"], json.dumps({"totals": tot, "median_factors": med}, indent=1))
    (HERE / "node_chain.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
