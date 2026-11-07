"""levin-phase-floor 07, the fourth stop: section 2 on 06d's code with 07's section 1.

``calibrate.py``, as written, does not finish its fourth (R) case at Phi_w = 2 pi -- c = 1e6,
x_s = 0.7236067977499790, omega = 0 -- in 25 minutes. This script records why, on the tree rebased
onto 06d (prompt section 9 point 7):

1. **The parent on every (R) case.** 06d's driver (``git show e5764a6:...``), as ``calibrate.py``
   runs it (rtol 1e-10, atol 1e-14 |ref|, defaults otherwise): honest, converged, abserr against
   the request max(atol, rtol |value|), the summed round-off floor of its leaves, leaves, depth,
   floor-limited leaves, wall time. Whether a case can converge at all at section 2's tolerances.
2. **The stuck case at bounded depth,** under 07, at each Phi_w rung and depth_max = 6, 9, 12: the
   leaves, the window leaves, the floor-limited leaves; abserr split between window leaves and the
   rest, and between floors and residuals; the Levin leaves outside the windows whose residual is
   above their floor (not floor-limited, so refinable); wall time.

Writes ``out/stop4_diagnose.json``.
"""

import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import reference as R  # noqa: E402
from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))
import AdaptiveLevin.levin_quadrature as lq  # noqa: E402

PARENT_SHA = "e5764a6"


def load_parent():
    source = subprocess.run(
        ["git", "show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"],
        cwd=MAIN,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    module = types.ModuleType("levin_quadrature_parent")
    exec(compile(source, "levin_quadrature_parent.py", "exec"), module.__dict__)
    return module


def amplitudes():
    return [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)]


def phase(c, x_s, omega):
    return {
        "theta": lambda x: omega * x + 0.5 * c * (x - x_s) ** 2,
        "theta_deriv": lambda x: omega + c * (x - x_s),
    }


def summarise(res, ref, wall):
    regions = res["regions"]
    window = [g for g in regions if g.type == "window"]
    other = [g for g in regions if g.type != "window"]
    flat_levin = [
        g
        for g in other
        if g.type == "Levin" and not g.phase_limited and g.abserr > g.phase_err
    ]
    request = max(1e-14 * abs(ref), 1e-10 * abs(res["value"]))
    return {
        "wall_s": wall,
        "true_error": abs(res["value"] - ref),
        "abserr": res["abserr"],
        "request": request,
        "abserr_over_request": res["abserr"] / request,
        "honest": abs(res["value"] - ref) <= res["abserr"],
        "converged": res["converged"],
        "leaves": len(regions),
        "max_depth": res["max_depth"],
        "floor_limited": res["num_phase_limited_regions"],
        "sum_floor_all": sum(g.phase_err for g in regions if g.phase_err is not None),
        "window_leaves": len(window),
        "window_sum_total_err": sum(g.total_err for g in window),
        "window_sum_floor": sum(g.phase_err for g in window),
        "other_sum_total_err": sum(g.total_err for g in other),
        "other_floor_limited": sum(1 for g in other if g.phase_limited),
        "levin_residual_above_floor": len(flat_levin),
        "levin_residual_above_floor_sum": sum(g.abserr for g in flat_levin),
    }


def main():
    log = lambda s: print(s, flush=True)  # noqa: E731
    parent = load_parent()
    out = {"header": header(Path(__file__), "07: the fourth stop"), "parent_R": {}}
    for case in R.cases():
        if case["family"] != "R":
            continue
        ref = float(R.closed_R(case["c"], case["x_s"], case["omega"]))
        t0 = time.perf_counter()
        res = parent.adaptive_levin_sincos(
            (0.0, 1.0),
            amplitudes(),
            phase(case["c"], case["x_s"], case["omega"]),
            atol=1e-14 * abs(ref),
            rtol=1e-10,
        )
        row = summarise(res, ref, time.perf_counter() - t0)
        name = f"c={case['c']:.0e} x_s={case['x_s']:.4f} omega={case['omega']:g}"
        out["parent_R"][name] = row
        log(
            f"parent {name}: honest {row['honest']} converged {row['converged']} "
            f"abserr/request {row['abserr_over_request']:.3g} leaves {row['leaves']}"
        )

    c, x_s = 1.0e6, 0.7236067977499790
    ref = float(R.closed_R(c, x_s, 0.0))
    out["stuck_case"] = {"c": c, "x_s": x_s, "omega": 0.0, "rungs": {}}
    saved = lq._LEVIN_WINDOW_HALF_PHASE
    try:
        for k in (2.0, 8.0, 32.0, 128.0, 512.0):
            lq._LEVIN_WINDOW_HALF_PHASE = k * math.pi
            rows = {}
            for depth_max in (6, 9, 12):
                t0 = time.perf_counter()
                res = lq._adaptive_levin(
                    (0.0, 1.0),
                    amplitudes(),
                    lq._Basis_SinCos(phase(c, x_s, 0.0)),
                    atol=1e-14 * abs(ref),
                    rtol=1e-10,
                    depth_max=depth_max,
                )
                rows[str(depth_max)] = summarise(res, ref, time.perf_counter() - t0)
                r = rows[str(depth_max)]
                log(
                    f"Phi_w {k:g}pi depth_max {depth_max}: {r['wall_s']:.1f}s leaves "
                    f"{r['leaves']} window {r['window_leaves']} abserr/request "
                    f"{r['abserr_over_request']:.3g} window err {r['window_sum_total_err']:.3g} "
                    f"other err {r['other_sum_total_err']:.3g} flat Levin "
                    f"{r['levin_residual_above_floor']}"
                )
            out["stuck_case"]["rungs"][f"{k:g}pi"] = rows
    finally:
        lq._LEVIN_WINDOW_HALF_PHASE = saved
    path = OUT / "stop4_diagnose.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
