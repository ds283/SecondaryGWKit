"""levin-phase-floor 07, the fourth stop, continued: the LiouvilleGreen draw that does not finish.

``lg_repeat.py`` (seeds 0-19 against ``e5764a6``) ran for more than four hours inside seed 0 under
the rebased tree and was ended by hand. Seed 0's first test is
``test_3bessel_analytic.Test3BesselAnalytic.test_JJJ``; its Levin calls 1-14 finish in at most 6 s
each, and call 15 does not finish. This script reproduces the draw (``random.seed(0)``, that test
alone, which draws first in the module) and, at call 15, records instead of finishing it:
- 06d's driver (``git show e5764a6:...``) and the rebased tree's, each at depth_max 8 and 12:
  leaves, converged, abserr, floor-limited leaves, wall time; and 06d's at the call's own
  depth_max;
- under the rebased tree, at depth_max 6, 8 and 10: the located points, the window leaves and their
  summed error, the other leaves and theirs, and the call's request max(atol, rtol |value|).
Then the test is abandoned (an exception raised from the wrapper).

Writes ``out/stop4_lg_seed0.json``.
"""

import json
import random
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))
import AdaptiveLevin.levin_quadrature as lq  # noqa: E402

PARENT_SHA = "e5764a6"
STUCK_CALL = 15


class _Abandon(Exception):
    pass


def main():
    source = subprocess.run(
        ["git", "show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"],
        cwd=MAIN,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    parent = types.ModuleType("levin_quadrature_parent")
    exec(compile(source, "levin_quadrature_parent.py", "exec"), parent.__dict__)
    new = lq._adaptive_levin
    out = {"header": header(Path(__file__), "07: the LiouvilleGreen draw, seed 0")}
    count = [0]

    def wrapped(x_span, f, basis, *args, **kwargs):
        count[0] += 1
        if count[0] != STUCK_CALL:
            return new(x_span, f, basis, *args, **kwargs)
        rec = {"x_span": [float(x_span[0]), float(x_span[1])], "kwargs": {}}
        rec["kwargs"] = {
            k: v for k, v in kwargs.items() if isinstance(v, (int, float, str))
        }
        rec["by_driver"] = []
        for depth_max in (8, 12):
            for name, driver in (("06d", parent._adaptive_levin), ("07", new)):
                kw = dict(kwargs)
                kw["depth_max"] = depth_max
                t0 = time.perf_counter()
                r = driver(x_span, f, basis, *args, **kw)
                rec["by_driver"].append(
                    {
                        "driver": name,
                        "depth_max": depth_max,
                        "wall_s": time.perf_counter() - t0,
                        "leaves": r["num_regions"],
                        "converged": r["converged"],
                        "abserr": r["abserr"],
                        "floor_limited": r["num_phase_limited_regions"],
                        "guarded": r.get("num_guarded_regions"),
                    }
                )
                print(json.dumps(rec["by_driver"][-1]), flush=True)
        t0 = time.perf_counter()
        r = parent._adaptive_levin(x_span, f, basis, *args, **kwargs)
        rec["parent_full"] = {
            "wall_s": time.perf_counter() - t0,
            "leaves": r["num_regions"],
            "converged": r["converged"],
            "abserr": r["abserr"],
            "max_depth": r["max_depth"],
        }
        print(json.dumps(rec["parent_full"]), flush=True)
        rec["windows"] = []
        for depth_max in (6, 8, 10):
            kw = dict(kwargs)
            kw["depth_max"] = depth_max
            r = new(x_span, f, basis, *args, **kw)
            window = [g for g in r["regions"] if g.type == "window"]
            other = [g for g in r["regions"] if g.type != "window"]
            rec["windows"].append(
                {
                    "depth_max": depth_max,
                    "points": [
                        (p["kind"], p["x"], p["window"], p["window_pieces"])
                        for p in r["stationary_points"]
                    ],
                    "window_leaves": len(window),
                    "window_sum_total_err": sum(g.total_err for g in window),
                    "other_leaves": len(other),
                    "other_sum_total_err": sum(g.total_err for g in other),
                    "request": max(
                        kwargs.get("atol", 0.0),
                        kwargs.get("rtol", 0.0) * abs(r["value"]),
                    ),
                }
            )
            print(json.dumps(rec["windows"][-1], default=str), flush=True)
        out["stuck_call"] = rec
        raise _Abandon()

    lq._adaptive_levin = wrapped
    random.seed(0)
    try:
        unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(
            unittest.TestLoader().loadTestsFromName(
                "LiouvilleGreen.tests.test_3bessel_analytic.Test3BesselAnalytic.test_JJJ"
            )
        )
    finally:
        lq._adaptive_levin = new
    out["calls_before_the_stuck_one"] = STUCK_CALL - 1
    path = OUT / "stop4_lg_seed0.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
