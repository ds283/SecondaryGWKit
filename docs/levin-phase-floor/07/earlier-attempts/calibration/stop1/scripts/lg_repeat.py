"""levin-phase-floor 07, the orchestrator's item 1, continued: does any draw move a test?

``LiouvilleGreen/tests/test_3bessel_analytic.py`` draws its wavenumbers from ``random.uniform``
unseeded on every run (its module docstring), so which of its Levin calls hold a stationary point
varies from run to run: ``lg_location.py`` saw 21 such calls in two tests on one run and 19 in three
on the next. This script runs that module over seeds 0..N_SEEDS-1, each seed twice:
- with the new driver, counting the calls that locate a point;
- with the dispatch SHA's driver (``git show 93d5cb1:AdaptiveLevin/levin_quadrature.py``, its
  ``_adaptive_levin`` put in place of the new one at the module attribute ``adaptive_levin_sincos``
  calls through), on the same draw.
``random.seed(seed)`` is set before each run, so both see the same wavenumbers. Recorded per seed
and driver: tests run, failures and errors by test id, and (new driver) the located calls.

Writes ``out/lg_repeat.json``.
"""

import json
import random
import subprocess
import sys
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))
import AdaptiveLevin.levin_quadrature as lq  # noqa: E402

N_SEEDS = 20
PARENT_SHA = "93d5cb1"
MODULE = "LiouvilleGreen.tests.test_3bessel_analytic"

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
LOCATED = []


def counting(*args, **kwargs):
    result = NEW(*args, **kwargs)
    if result["stationary_points"]:
        LOCATED.append(
            {
                "x_span": [float(args[0][0]), float(args[0][1])],
                "points": [(p["kind"], p["x"]) for p in result["stationary_points"]],
            }
        )
    return result


def run_once(seed, driver):
    random.seed(seed)
    LOCATED.clear()
    lq._adaptive_levin = counting if driver == "new" else parent._adaptive_levin
    try:
        suite = unittest.TestLoader().loadTestsFromName(MODULE)
        outcome = unittest.TextTestRunner(
            stream=open("/dev/null", "w"), verbosity=0
        ).run(suite)
    finally:
        lq._adaptive_levin = NEW
    return {
        "tests_run": outcome.testsRun,
        "failures": [t.id() for t, _ in outcome.failures],
        "errors": [t.id() for t, _ in outcome.errors],
        "skipped": len(outcome.skipped),
        "located_calls": len(LOCATED) if driver == "new" else None,
        "located": list(LOCATED) if driver == "new" else None,
    }


def main():
    out = {
        "header": header(
            Path(__file__), "07: LiouvilleGreen analytic tests over draws"
        ),
        "n_seeds": N_SEEDS,
        "parent_sha": PARENT_SHA,
        "module": MODULE,
        "seeds": {},
    }
    for seed in range(N_SEEDS):
        new = run_once(seed, "new")
        old = run_once(seed, "parent")
        out["seeds"][seed] = {"new": new, "parent": old}
        print(
            f"seed {seed}: new failures {new['failures'] + new['errors']} "
            f"located calls {new['located_calls']}; parent failures "
            f"{old['failures'] + old['errors']}",
            flush=True,
        )
    out["seeds_with_a_new_failure"] = [
        s for s, r in out["seeds"].items() if r["new"]["failures"] or r["new"]["errors"]
    ]
    out["seeds_with_a_parent_failure"] = [
        s
        for s, r in out["seeds"].items()
        if r["parent"]["failures"] or r["parent"]["errors"]
    ]
    out["seeds_locating"] = sum(
        1 for r in out["seeds"].values() if r["new"]["located_calls"]
    )
    out["located_calls_total"] = sum(
        r["new"]["located_calls"] for r in out["seeds"].values()
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "seeds_with_a_new_failure",
                    "seeds_with_a_parent_failure",
                    "seeds_locating",
                    "located_calls_total",
                )
            }
        )
    )
    path = OUT / "lg_repeat.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
