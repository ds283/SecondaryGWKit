"""Measure 1 and 2: 06d's driver on the three (R) cases, per final leaf and in total.

Per leaf: span, h, depth, branch, declared value and error, every floor term, the exact leaf
integral (closed form at MP_DPS), the same-scheme mp value, and so round-off and truncation.
Writes driver_run.json.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/docs/levin-phase-floor/07/calibration/scripts")

import math  # noqa: E402

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import reference as R  # noqa: E402  (07's, read only: closed_R and contour_R)

lq = C.lq


def main():
    out = {"header": C.header(__file__), "cases": {}}
    for case in C.CASES:
        t0 = time.perf_counter()
        ref_closed = R.closed_R(case["c"], case["x_s"], case["omega"])
        ref_contour = R.contour_R(case["c"], case["x_s"], case["omega"])
        ref = float(ref_closed)
        atol = C.ATOL_FACTOR * abs(ref)
        # record the driver's own solve of every span, to check the recomputed leaves against
        seen = {}
        orig = lq._adaptive_levin_subregion

        def spy(span, *a, **k):
            d = orig(span, *a, **k)
            seen[(span[0], span[1])] = (d["value"], d["phase_err"])
            return d

        lq._adaptive_levin_subregion = spy
        try:
            res = lq.adaptive_levin_sincos(
                (0.0, 1.0), C.amplitudes(), C.phase(case), atol=atol, rtol=C.RTOL
            )
        finally:
            lq._adaptive_levin_subregion = orig
        request = max(atol, C.RTOL * abs(res["value"]))
        xr = C.x_root(case)
        leaves = []
        for reg in res["regions"]:
            span = (reg.start, reg.end)
            row = C.analyse_leaf(case, span)
            row.update(
                {
                    "depth": reg.depth,
                    "type": reg.type,
                    "residual": reg.abserr,
                    "phase_err_region": reg.phase_err,
                    "truncation_term": reg.abserr_truncation,
                    "declared_error": reg.total_err,
                    "phase_limited": reg.phase_limited,
                    "trusted": reg.trusted,
                    "holds_x_r": bool(reg.start <= xr <= reg.end),
                }
            )
            assert row["floor_code"] == reg.phase_err, (row["floor_code"], reg.phase_err)
            assert (row["value"], row["floor_code"]) == seen[span], (row["value"], seen[span])
            leaves.append(row)
        with mp.workdps(C.MP_DPS):
            vsum = math.fsum(r["value"] for r in leaves)
            exact_sum = mp.fsum(mp.mpf(r["exact"]) for r in leaves)
            total_true_err = float(mp.mpf(res["value"]) - ref_closed)
        terms = sorted({k for r in leaves for k in r["terms"]})
        summ = {
            "ref_closed": mp.nstr(ref_closed, 30),
            "ref_closed_minus_contour": float(ref_closed - ref_contour),
            "sum_exact_leaves_minus_ref": float(exact_sum - ref_closed),
            "value": res["value"],
            "value_recomputed_fsum": vsum,
            "abserr": res["abserr"],
            "request": request,
            "true_error": total_true_err,
            "converged": res["converged"],
            "num_regions": res["num_regions"],
            "num_cc": sum(1 for r in leaves if r["kind"] == "CC"),
            "num_floor_limited": sum(1 for r in leaves if r["phase_limited"]),
            "max_depth": res["max_depth"],
            "x_r": xr,
            "sum_declared_error": math.fsum(r["declared_error"] for r in leaves),
            "sum_floor": math.fsum(r["floor_code"] for r in leaves),
            "sum_terms": {
                t: math.fsum(r["terms"].get(t, 0.0) for r in leaves) for t in terms
            },
            "sum_abs_roundoff": math.fsum(abs(r["roundoff"]) for r in leaves),
            "abs_sum_roundoff": abs(math.fsum(r["roundoff"] for r in leaves)),
            "sum_abs_truncation": math.fsum(abs(r["truncation"]) for r in leaves),
            "abs_sum_truncation": abs(math.fsum(r["truncation"] for r in leaves)),
            "sum_abs_actual": math.fsum(abs(r["actual_error"]) for r in leaves),
            "abs_sum_actual": abs(math.fsum(r["actual_error"] for r in leaves)),
            "levin_endpoint_part_sum_abs": math.fsum(
                abs(r.get("roundoff_endpoint_phase_part", 0.0)) for r in leaves
            ),
            "levin_endpoint_part_abs_sum": abs(
                math.fsum(r.get("roundoff_endpoint_phase_part", 0.0) for r in leaves)
            ),
            "rest_part_sum_abs": math.fsum(
                abs(r.get("roundoff_rest_part", 0.0)) for r in leaves
            ),
            "rest_part_abs_sum": abs(
                math.fsum(r.get("roundoff_rest_part", 0.0) for r in leaves)
            ),
            "cc_node_part_sum_abs": math.fsum(
                abs(r.get("roundoff_node_part", 0.0)) for r in leaves
            ),
            "leaves_roundoff_above_floor": sum(
                1 for r in leaves if abs(r["roundoff"]) > r["floor_code"]
            ),
            "leaves_actual_above_declared": sum(
                1 for r in leaves if abs(r["actual_error"]) > r["declared_error"]
            ),
            "grid_ends_not_exact": sum(
                1 for r in leaves if r.get("grid_ends_exact") is False
            ),
            "wall_s": time.perf_counter() - t0,
        }
        out["cases"][case["name"]] = {"case": case, "summary": summ, "leaves": leaves}
        print(case["name"], json.dumps(summ, indent=1, default=str), flush=True)
    path = HERE / "driver_run.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
