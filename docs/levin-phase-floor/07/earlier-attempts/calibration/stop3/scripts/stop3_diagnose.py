"""levin-phase-floor 07, the third stop: section 2 cannot run on 06c's code with 07's section 1.

Three measurements, on the tree rebased onto 06c (prompt section 9 point 6):
1. **The case section 2 stops on.** (R), c = 1e6, x_s = 0.7236067977499790, omega = 0, at
   Phi_w = 2 pi, run exactly as ``calibrate.py`` runs it (rtol 1e-10, atol 1e-14 |ref|), with
   depth_max = 0, 1, 2, 4, 6, 8, 10 in turn. For each: the window leaves' summed eq. (151) floor,
   summed nested-pair estimate and summed total_err; the window leaf with the largest error, its
   floor and its span; the other leaves' summed error; the true error.
2. **The full call** (depth_max 20, the default), timed, with the exception it raises.
3. **The LiouvilleGreen draw that moved:** ``test_3bessel_analytic.Test3BesselAnalytic.test_JJJ``
   on seed 0 (``random.seed(0)`` and then that test alone, which draws first in the module), under
   the new driver and under 06c's (``git show ad63e8a:...``), each timed, with every Levin call
   that locates a point recorded.

Writes ``out/stop3_diagnose.json``.
"""

import json
import math
import random
import subprocess
import sys
import time
import traceback
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import reference as R  # noqa: E402
from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))
import AdaptiveLevin.levin_quadrature as lq  # noqa: E402

PARENT_SHA = "ad63e8a"
C, XS = 1.0e6, 0.7236067977499790


def case_call(depth_max):
    ref = float(R.closed_R(C, XS, 0.0))
    f = [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)]
    theta = {
        "theta": lambda x: 0.5 * C * (x - XS) ** 2,
        "theta_deriv": lambda x: C * (x - XS),
    }
    t0 = time.perf_counter()
    r = lq.adaptive_levin_sincos(
        (0.0, 1.0), f, theta, atol=1e-14 * abs(ref), rtol=1e-10, depth_max=depth_max
    )
    wall = time.perf_counter() - t0
    window = [g for g in r["regions"] if g.type == "window"]
    other = [g for g in r["regions"] if g.type != "window"]
    worst = max(window, key=lambda g: g.total_err)
    return {
        "depth_max": depth_max,
        "wall_s": wall,
        "request": max(1e-14 * abs(ref), 1e-10 * abs(r["value"])),
        "true_error": abs(r["value"] - ref),
        "abserr": r["abserr"],
        "converged": r["converged"],
        "window_leaves": len(window),
        "window_sum_floor": sum(g.phase_err for g in window),
        "window_sum_nested_pair": sum(g.abserr for g in window),
        "window_sum_total_err": sum(g.total_err for g in window),
        "worst_window_leaf": {
            "start": worst.start,
            "end": worst.end,
            "depth": worst.depth,
            "floor": worst.phase_err,
            "nested_pair": worst.abserr,
            "holds_x_s": min(worst.start, worst.end)
            <= XS
            <= max(worst.start, worst.end),
        },
        "other_leaves": len(other),
        "other_sum_total_err": sum(g.total_err for g in other),
        "num_untrusted_regions": r["num_untrusted_regions"],
        "num_phase_limited_regions": r["num_phase_limited_regions"],
    }


def lg_seed0(driver):
    calls = []
    original = lq._adaptive_levin

    def wrapped(*args, **kwargs):
        result = driver(*args, **kwargs)
        if result.get("stationary_points"):
            calls.append(
                {
                    "x_span": [float(args[0][0]), float(args[0][1])],
                    "points": [
                        (p["kind"], p["x"]) for p in result["stationary_points"]
                    ],
                    "value": result["value"],
                    "abserr": result["abserr"],
                    "converged": result["converged"],
                    "regions": result["num_regions"],
                    "window_pieces": result["num_window_pieces"],
                    "max_depth": result["max_depth"],
                }
            )
        return result

    lq._adaptive_levin = wrapped
    random.seed(0)
    t0 = time.perf_counter()
    try:
        suite = unittest.TestLoader().loadTestsFromName(
            "LiouvilleGreen.tests.test_3bessel_analytic.Test3BesselAnalytic.test_JJJ"
        )
        outcome = unittest.TextTestRunner(
            stream=open("/dev/null", "w"), verbosity=0
        ).run(suite)
    finally:
        lq._adaptive_levin = original
    return {
        "wall_s": time.perf_counter() - t0,
        "failures": [msg for _, msg in outcome.failures],
        "errors": [msg for _, msg in outcome.errors],
        "located_calls": calls,
    }


def main():
    out = {"header": header(Path(__file__), "07: the third stop")}
    saved = lq._LEVIN_WINDOW_HALF_PHASE
    try:
        lq._LEVIN_WINDOW_HALF_PHASE = 2.0 * math.pi
        out["by_depth_max"] = [case_call(d) for d in (0, 1, 2, 4, 6, 8, 10)]
        for row in out["by_depth_max"]:
            print(json.dumps(row), flush=True)
        t0 = time.perf_counter()
        try:
            full = case_call(20)
            out["full_call"] = full
        except Exception as e:  # recorded, not handled
            out["full_call"] = {
                "exception": repr(e),
                "traceback_tail": traceback.format_exc().splitlines()[-6:],
                "wall_s": time.perf_counter() - t0,
            }
        print(json.dumps(out["full_call"], default=str), flush=True)
    finally:
        lq._LEVIN_WINDOW_HALF_PHASE = saved

    source = subprocess.run(
        ["git", "show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"],
        cwd=MAIN,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    parent = types.ModuleType("levin_quadrature_parent")
    exec(compile(source, "levin_quadrature_parent.py", "exec"), parent.__dict__)
    out["lg_seed0_test_JJJ"] = {
        "new": lg_seed0(lq._adaptive_levin),
        "parent": lg_seed0(parent._adaptive_levin),
    }
    for k, v in out["lg_seed0_test_JJJ"].items():
        print(k, v["wall_s"], [m.splitlines()[-1] for m in v["failures"]], flush=True)
    path = OUT / "stop3_diagnose.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
