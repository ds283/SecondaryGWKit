"""Questions 1 and 2: the leaves outside the window, leaf by leaf, for one case at the given Phi_w rungs
and forced depth_max values.

Usage: q12_outside.py CASE PHI1,PHI2,... D1,D2,... [--theta-mod] [--tag TAG]
  CASE is a key of lc.CASES; PHI are Phi_w / pi; D are forced depth_max.

For each run: the call's value, abserr, true error, request; the window's summed total_err; every
leaf outside the window with its position in phase from x_r (near and far edge), theta' at the near
edge, type, depth, abserr, floor, truncation, the trust check's delta1/delta2/r/declared resolution,
the component that sets its declared error (lc.component), and its true error against the closed
form; and, for the leaves carrying 99% of the outside declared error (at most 40), their halves'
values and true errors from the driver's own recorded solves: |V_P - exact|, |V_L + V_R - exact|.

Writes out/q12_<CASE>[_TAG].json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mpmath as mp  # noqa: E402

import lc  # noqa: E402


def main():
    case_key = sys.argv[1]
    rungs = [float(v) for v in sys.argv[2].split(",")]
    depths = [int(v) for v in sys.argv[3].split(",")]
    theta_mod = "--theta-mod" in sys.argv
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else None
    case = lc.CASES[case_key]
    with mp.workdps(lc.MP_DPS):
        ref = lc.closed_leaf(case, 0.0, 1.0)
    ref_f = float(ref)
    atol = 1e-14 * abs(ref_f)
    out = {
        "header": lc.header(__file__),
        "case": case,
        "x_root": lc.x_root(case),
        "reference": mp.nstr(ref, 30),
        "atol": atol,
        "rtol": lc.RTOL,
        "theta_mod_2pi_supplied": theta_mod,
        "runs": {},
    }
    name = f"q12_{case_key}" + (f"_{tag}" if tag else "")
    path = lc.OUT / f"{name}.json"
    for k in rungs:
        for dm in depths:
            t0 = time.perf_counter()
            res, basis = lc.run_driver(case, k, dm, theta_mod=theta_mod, atol=atol)
            wall = time.perf_counter() - t0
            regions = res["regions"]
            window = [g for g in regions if g.type == "window"]
            other = [g for g in regions if g.type != "window"]
            with mp.workdps(lc.MP_DPS):
                true_err = float(mp.mpf(res["value"]) - ref)
            request = max(atol, lc.RTOL * abs(res["value"]))
            leaves = [lc.outside_leaf_record(case, g) for g in other]
            # carriers: largest total_err first, until 99% of the outside sum, at most 40
            order = sorted(range(len(other)), key=lambda i: -leaves[i]["total_err"])
            tot = sum(rec["total_err"] for rec in leaves)
            acc = 0.0
            carriers = []
            for i in order:
                if acc >= 0.99 * tot or len(carriers) >= 40:
                    break
                acc += leaves[i]["total_err"]
                carriers.append(i)
            for i in carriers:
                rec = leaves[i]
                a, b = rec["a"], rec["b"]
                c = (b + a) / 2.0
                L = lc.REC["solves"].get((a, c))
                R = lc.REC["solves"].get((c, b))
                rec["carrier"] = True
                if L is not None and R is not None:
                    with mp.workdps(lc.MP_DPS):
                        exL = lc.closed_leaf(case, a, c)
                        exR = lc.closed_leaf(case, c, b)
                        ex = exL + exR
                        rec["halves"] = {
                            "L_value": L["value"],
                            "R_value": R["value"],
                            "L_true_error": float(mp.mpf(L["value"]) - exL),
                            "R_true_error": float(mp.mpf(R["value"]) - exR),
                            "halves_sum_true_error": float(
                                mp.mpf(L["value"]) + mp.mpf(R["value"]) - ex
                            ),
                            "L_floor": L["phase_err"],
                            "R_floor": R["phase_err"],
                            "L_is_direct": L["is_direct"],
                            "R_is_direct": R["is_direct"],
                            "L_trunc": L["abserr_truncation"],
                            "R_trunc": R["abserr_truncation"],
                            "L_p_ratios": L["p_ratios"],
                            "R_p_ratios": R["p_ratios"],
                        }
            comp_sum = {}
            for rec in leaves:
                comp_sum[rec["component"]] = comp_sum.get(rec["component"], 0.0) + rec[
                    "total_err"
                ]
            run = {
                "phi_w_over_pi": k,
                "depth_max": dm,
                "wall_s": wall,
                "value": res["value"],
                "abserr": res["abserr"],
                "true_error": true_err,
                "request": request,
                "converged": res["converged"],
                "max_depth": res["max_depth"],
                "leaves": len(regions),
                "window_leaves": len(window),
                "window_sum_total_err": sum(g.total_err for g in window),
                "window_sum_value": sum(g._rec_value for g in window),
                "outside_leaves": len(other),
                "outside_sum_total_err": tot,
                "outside_sum_true_error": sum(r["true_error"] for r in leaves),
                "outside_sum_abs_true_error": sum(abs(r["true_error"]) for r in leaves),
                "outside_by_component": comp_sum,
                "num_untrusted": res["num_untrusted_regions"],
                "num_phase_limited": res["num_phase_limited_regions"],
                "num_guarded": res["num_guarded_regions"],
                "stationary_points": res["stationary_points"],
                "outside": leaves,
            }
            with mp.workdps(lc.MP_DPS):
                pts = res["stationary_points"]
                if pts:
                    lo, hi = pts[0]["window"]
                    exw = lc.closed_leaf(case, lo, hi)
                    run["window_true_error"] = float(mp.mpf(run["window_sum_value"]) - exw)
            out["runs"][f"{k:g}pi_d{dm}"] = run
            print(
                f"{case_key} {k:g}pi d{dm}: {wall:.1f}s abserr {res['abserr']:.3g} true "
                f"{true_err:.3g} window {run['window_sum_total_err']:.3g} outside {tot:.3g} "
                f"(true sum {run['outside_sum_true_error']:.3g}) comps "
                + ", ".join(f"{c}={v:.2g}" for c, v in comp_sum.items()),
                flush=True,
            )
            path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
