"""
levin-phase-floor prompt 04, step 5 (diagnostic, not a gate, not scored): what is the residual
jitter of the composed phase under arm X?

Hypothesis tested: it is the rounding of the abscissa, z = expm1(x) to the nearest double inside
PrimitivePhase._get_x, carried into the phase as (d theta/du) * delta_z / (1 + z).  For 03's first
three sampled depth-20 regions of a few scored groups: the deg-4 fit residual of the composed
theta_mod_2pi under X (03's estimator, 65 points), beside the same residual of the predicted
series sum_i s_i (d theta_i/du) (z_double - z_exact)/(1 + z), with z_exact = expm1(x) in mpmath,
and their correlation.  Ray-free, no Levin call, route (a) as in check_phases.py.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import RUN03, header  # noqa: E402

CASES = [("R3", "Gr", "G-r"), ("R1", "Gqr", "G+q+r"), ("R4", "Gr", "G-r"), ("R6", "G", "G")]
ROWS = {"R1": (2, 37527), "R3": (1, 37670), "R4": (2, 37720), "R6": (2, 34728)}


def resid(x, th):
    th = np.unwrap(th)
    t = (2.0 * x - (x[0] + x[-1])) / (x[-1] - x[0])
    c = np.polynomial.polynomial.polyfit(t, th, 4)
    return th - np.polynomial.polynomial.polyval(t, c)


def main():
    import importlib
    from math import log

    import mpmath
    from harness import _ray_copy, close_pool, load_record, open_readonly_pool, standin_ray
    from rebuild import PRIMARY, build_context, rebuild_row
    import phases as PH
    from ComputeTargets.TkSourceFunctions import TkSourceFunctions
    from ComputeTargets.phase_groups import build_phase_groups

    QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")
    mpmath.mp.prec = 200
    for line in header(Path(__file__).resolve(), "abscissa-rounding diagnostic", inputs=[HERE / "phases.py"]):
        print(line)
    with standin_ray() as cluster:
        pool, _ = open_readonly_pool(PRIMARY, cluster)
        try:
            ctx = build_context(pool, log=lambda s: None)
            for name, regime_label, label in CASES:
                shard, serial = ROWS[name]
                rb = rebuild_row(ctx, pool, shard, serial, log=lambda s: None)
                payload = _ray_copy(rb["compute_payload"])
                model = _ray_copy(ctx["bg_model"])
                GkPolicy = payload["GkPolicy"]
                Gk_f = GkPolicy.functions
                Tq_f = TkSourceFunctions(model, rb["q"].k, payload["Tq_numeric"], payload["Tq_WKB"])
                Tr_f = TkSourceFunctions(model, rb["r"].k, payload["Tr_numeric"], payload["Tr_WKB"])
                part = QSI.build_partition(GkPolicy, Tq_f, Tr_f, payload["source"], rb["z_response"].z, ctx["z_source_integral_max_z"].z)
                item = [it for it in part["subintervals"] if it["method"] != "quad" and "".join(n for n, f in zip("Gqr", it["regime"]) if f) == regime_label][0]
                regime = tuple(item["regime"])
                groups = build_phase_groups(regime, Gk=Gk_f if regime[0] else Gk_f.numeric_Gk, Tq=item["Tq"], Tr=item["Tr"], model_functions=model.functions, w_background=model.functions.wBackground)
                g = [gg for gg in groups if gg.label == label][0]
                factors = PH.factor_list(regime, Gk_f.phase, item["Tq"], item["Tr"])
                signs = dict(zip("Gqr", g.signs))
                rec = load_record(RUN03 / f"row_{name}.pkl")
                call = [c for c in rec["levin_calls"] if c["group_label"] == label and c["parsed"]["regime"] == regime_label][0]
                d20 = [r for r in call["result"]["regions"] if r.depth == 20]
                idx = sorted(set(int(round(i)) for i in np.linspace(0, len(d20) - 1, min(20, len(d20)))))
                for i in idx[:3]:
                    r = d20[i]
                    x = np.linspace(r.start, r.end, 65)
                    with PH.install("X"):
                        thX = np.array([g.theta_mod_2pi(float(v)) for v in x])
                    pred = []
                    for v in x:
                        tot = 0.0
                        for fn, ph, clamp in factors:
                            s = signs[fn]
                            if s == 0:
                                continue
                            xc = clamp(float(v))
                            z_dbl = PH.pieces(ph, xc)[0]
                            z_ex = mpmath.expm1(mpmath.mpf(xc))
                            dz = float(mpmath.mpf(z_dbl) - z_ex)
                            tot += s * ph.theta_deriv(xc, x_is_log=True, log_derivative=True) * dz / (1.0 + z_dbl)
                        pred.append(tot)
                    pred = np.array(pred)
                    rX = resid(x, thX)
                    rP = resid(x, pred)
                    corr = float(np.corrcoef(rX, rP)[0, 1])
                    left = float(np.sqrt(np.mean((rX - rP) ** 2)))
                    print(f"{name} {regime_label} {label} region {i}: x {r.start:.6f}; RMS resid X {np.sqrt(np.mean(rX*rX)):.3g}; predicted from z rounding {np.sqrt(np.mean(rP*rP)):.3g}; corr {corr:.4f}; RMS(X - predicted) {left:.3g}")
        finally:
            close_pool(pool, cluster)


if __name__ == "__main__":
    main()
