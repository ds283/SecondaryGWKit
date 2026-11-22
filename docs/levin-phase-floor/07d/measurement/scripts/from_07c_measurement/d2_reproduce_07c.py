"""levin-phase-floor prompt 07c Part A, section 2.3: "06d's numbers and the audit's are the agent's to reproduce first, on
their own cases". New in 07c.

**The audit's case** (docs/adaptive-levin-audit-2026-09.md section 2.2, e09_cc.py): I = int_1^2 e^{-t}/(1+t) sin(W t) dt,
phase span W (the width is 1), at N = 13 (CC_13, CC_25), W in {0.5, 2, 4, 6, 10} pi. The audit's table gives CC_13's error,
CC_25's error and the estimate |CC_25 - CC_13|.

**06d's numbers** (the board's [06d-cc-nested-estimate-measures-the-coarse-rule], 06e's prompt section 0: "at a 6 pi span
CC_16's error is 2.7e-6 against CC_31's 5.4e-17; at 3 pi, 1.4e-10 against 3.4e-17"): their case is not named in the
records found; the same integrand is run at N = 16 (CC_16, CC_31) for W = 3 pi and 6 pi, and at N = 24 (the census
calls' order, CC_24 and CC_47) for every W.

Each rule is the module's own fallback, ``_adaptive_levin_subregion_cc`` (the tree's, imported unchanged), with
``_Basis_SinCos({"theta": W t, "theta_deriv": W})`` and f = [e^{-t}/(1+t), 0]; CC_N is w_N . g[::2] from its fine samples.
The reference is mpmath's quad at 40 digits. Every candidate of leafcc_07c.py is evaluated as there.

Writes out/d2/reproduce.json and prints the table. Usage: PYTHONPATH=<tree> python -B d2_reproduce_07c.py
"""

import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    import AdaptiveLevin.levin_quadrature as LQ

    out = ROOT / "out" / "d2" / "reproduce.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    mp.mp.dps = 40
    rows = []
    for N, Ws in ((13, (0.5, 2, 4, 6, 10)), (16, (3, 6)), (24, (0.5, 2, 3, 4, 6, 10))):
        for wpi in Ws:
            W = wpi * math.pi
            span = (1.0, 2.0)
            amp = lambda t: np.exp(-np.asarray(t)) / (1.0 + np.asarray(t))
            zero = lambda t: 0.0 * np.asarray(t)
            A = LQ._Basis_SinCos({"theta": lambda t, W=W: W * t, "theta_deriv": lambda t, W=W: W + 0.0 * t})
            f = [amp, zero]
            grid, D = LQ.chebyshev_matrices(span, N)
            _, tpc, _, _, V = A.build_Levin_data(grid, D, need_AmatT=False)
            d = LQ._adaptive_levin_subregion_cc(span, f, A, N, V, tpc, None, LQ._LazyUUID())
            fine = 2 * N - 1
            gf, _ = LQ.chebyshev_matrices(span, fine)
            g = np.array([sum(fn(x) * b for fn, b in zip(f, A.eval_basis_and_arguments(x)[0])) for x in gf])
            cfine = float(np.dot(LQ._cc_weights(span, fine), g))
            ccoarse = float(np.dot(LQ._cc_weights(span, N), g[::2]))
            g3x, _ = LQ.chebyshev_matrices(span, 4 * N - 3)
            g3 = np.array([sum(fn(x) * b for fn, b in zip(f, A.eval_basis_and_arguments(x)[0])) for x in g3x])
            c3 = float(np.dot(LQ._cc_weights(span, 4 * N - 3), g3))
            ref = mp.quad(lambda t: mp.e ** (-t) / (1 + t) * mp.sin(mp.mpf(W) * t), [1, 1.25, 1.5, 1.75, 2])
            n = fine
            j = np.arange(n)
            wts = np.ones(n)
            wts[0] = wts[-1] = 0.5
            ck = np.array([(2.0 / (n - 1)) * np.sum(wts * g * np.cos(np.pi * j * k / (n - 1))) for k in range(n)])
            ck[0] *= 0.5
            ck[-1] *= 0.5
            pair = abs(cfine - ccoarse)
            resasc = d["resasc"]
            row = {"N": N, "W_over_pi": wpi, "V_over_6pi": V / (6 * math.pi),
                   "value_equals_fallback": cfine == d["value"], "pair_equals_fallback": pair == d["abserr_direct"],
                   "err_CC_N": float(abs(mp.mpf(ccoarse) - ref)), "err_CC_2N-1": float(abs(mp.mpf(cfine) - ref)),
                   "err_CC_4N-3": float(abs(mp.mpf(c3) - ref)), "floor": d["phase_err"],
                   "cand_pair": pair, "cand_nest3": abs(c3 - cfine),
                   "cand_tail_sum4": 0.5 * 1.0 * float(np.sum(np.abs(ck[-4:]))),
                   "cand_tail_max4x2": 1.0 * float(np.max(np.abs(ck[-4:]))),
                   "cand_dqk21": resasc * min(1.0, (200.0 * pair / resasc) ** 1.5)}
            rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "module_sha256": hashlib.sha256(Path(LQ.__file__).read_bytes()).hexdigest(), "rows": rows}, indent=1))
    print("| N | W/pi | err CC_N | err CC_2N-1 | err CC_4N-3 | floor | pair | nest3 | tail_sum4 | tail_max4x2 | dqk21 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['N']} | {r['W_over_pi']} | {r['err_CC_N']:.2g} | {r['err_CC_2N-1']:.2g} | {r['err_CC_4N-3']:.2g} | {r['floor']:.2g} | "
              f"{r['cand_pair']:.2g} | {r['cand_nest3']:.2g} | {r['cand_tail_sum4']:.2g} | {r['cand_tail_max4x2']:.2g} | {r['cand_dqk21']:.2g} |")
    print("wrote", out)


if __name__ == "__main__":
    main()
