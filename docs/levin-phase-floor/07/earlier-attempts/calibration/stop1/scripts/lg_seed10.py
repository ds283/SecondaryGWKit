"""levin-phase-floor 07: the draw on which ``lg_repeat.py`` found a LiouvilleGreen test moving.

Seed 10 of ``lg_repeat.py`` fails ``test_3bessel_analytic.Test3BesselAnalytic.test_JJJ`` under the
new driver, at the provisional constants, and passes under the dispatch SHA's. The failing
assertion is the triangle (J2, J2, J0) at k = 3.3020154161157618, q = 0.8851248269422163,
s = 2.6512798622356306: relative error 2.57e-7 against the test's REL_TOLERANCE of 1e-7.

This script evaluates that one integral as the test does (``compute_3Bessel`` with ``quad_JJJ``),
under the parent, and under the new driver at each rung of section 2's Phi_w ladder, recording:
the value's error against the closed form, the declared abserr, and for each of the four Levin
calls its value, abserr, region count, and its located points with their windows. For the call
that locates a point, every accepted Levin region outside the window is listed with its abserr,
round-off floor, ``phase_limited`` flag and theta' at its two ends.

Writes ``out/lg_seed10.json``.
"""

import json
import math
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))
import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
import LiouvilleGreen.tests.test_3bessel_analytic as T  # noqa: E402
from LiouvilleGreen.three_bessel_integrals import quad_JJJ  # noqa: E402

PARENT_SHA = "93d5cb1"
K, Q, S = 3.3020154161157618, 0.8851248269422163, 2.6512798622356306
PHI_LADDER = tuple(k * math.pi for k in (2.0, 8.0, 32.0, 128.0, 512.0))

source = subprocess.run(
    ["git", "show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"],
    cwd=MAIN,
    capture_output=True,
    text=True,
    check=True,
).stdout
parent = types.ModuleType("levin_quadrature_parent")
exec(compile(source, "levin_quadrature_parent.py", "exec"), parent.__dict__)
NEW = lq._adaptive_levin


def evaluate(driver):
    J = T.Jintegrals[2]
    analytic = J.analytic(K, Q, S)
    calls = []

    def wrapped(*args, **kwargs):
        result = driver(*args, **kwargs)
        calls.append((args, result))
        return result

    lq._adaptive_levin = wrapped
    try:
        res = T.compute_3Bessel(
            quad_JJJ,
            J.mu,
            J.nu,
            J.sigma,
            K,
            Q,
            S,
            max_x=T.MAX_X,
            analytic_result=analytic,
            label="JJJ",
            timestamp=datetime.now(),
        )
    finally:
        lq._adaptive_levin = NEW
    rec = {
        "value": res.value,
        "analytic": analytic,
        "error": abs(res.value - analytic),
        "relative_error": abs(res.value - analytic) / abs(analytic),
        "abserr": res.abserr,
        "test_passes": abs(res.value - analytic) / abs(analytic) < T.REL_TOLERANCE
        or abs(res.value - analytic) < T.ABS_TOLERANCE,
        "calls": [],
    }
    for args, r in calls:
        call = {
            "x_span": list(args[0]),
            "value": r["value"],
            "abserr": r["abserr"],
            "regions": r["num_regions"],
            "max_depth": r["max_depth"],
            "num_phase_limited_regions": r["num_phase_limited_regions"],
        }
        points = r.get("stationary_points", [])
        if points:
            rate = args[2].independent_phase_rates()[0]
            call["points"] = [
                {
                    "x": p["x"],
                    "kind": p["kind"],
                    "window": p["window"],
                    "window_pieces": p["window_pieces"],
                }
                for p in points
            ]
            call["outer_levin_regions"] = [
                {
                    "start": reg.start,
                    "end": reg.end,
                    "depth": reg.depth,
                    "abserr": reg.abserr,
                    "phase_err": reg.phase_err,
                    "phase_limited": reg.phase_limited,
                    "theta_deriv_start": float(rate(reg.start)),
                    "theta_deriv_end": float(rate(reg.end)),
                }
                for reg in r["regions"]
                if reg.type == "Levin"
            ]
        rec["calls"].append(call)
    return rec


def main():
    out = {
        "header": header(Path(__file__), "07: the LiouvilleGreen draw that moved"),
        "k_q_s": [K, Q, S],
        "rel_tolerance": T.REL_TOLERANCE,
        "abs_tolerance": T.ABS_TOLERANCE,
        "parent": evaluate(parent._adaptive_levin),
        "new": {},
    }
    saved = lq._LEVIN_WINDOW_HALF_PHASE
    try:
        for phi in PHI_LADDER:
            lq._LEVIN_WINDOW_HALF_PHASE = phi
            out["new"][f"{phi / math.pi:g}pi"] = evaluate(NEW)
    finally:
        lq._LEVIN_WINDOW_HALF_PHASE = saved
    p = out["parent"]
    print(
        f"parent: rel err {p['relative_error']:.3g} abserr {p['abserr']:.3g} passes {p['test_passes']}"
    )
    for rung, r in out["new"].items():
        outer = [
            (
                round(o["start"], 3),
                round(o["end"], 3),
                o["depth"],
                o["phase_limited"],
                f"{o['phase_err']:.2g}",
                f"{o['theta_deriv_start']:.3g}",
            )
            for c in r["calls"]
            for o in c.get("outer_levin_regions", [])
        ]
        print(
            f"Phi_w {rung}: rel err {r['relative_error']:.3g} abserr {r['abserr']:.3g} "
            f"passes {r['test_passes']} outer Levin {outer[:4]}"
        )
    path = OUT / "lg_seed10.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
