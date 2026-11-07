"""levin-phase-floor 07, the orchestrator's item 1: what the stationary-point location does on the
Levin calls ``LiouvilleGreen/tests/`` makes.

``LiouvilleGreen/three_bessel_integrals.py`` calls ``adaptive_levin_sincos`` once per
sum-and-difference group, with ``theta_deriv`` supplied (``BesselPhaseGroup.levin_theta()``), so
the location runs on every one of its calls.

The script runs the whole ``LiouvilleGreen/tests/`` root in this process, exactly as
``python -m unittest discover -s LiouvilleGreen/tests -t .`` does, with one change: the module
attribute ``AdaptiveLevin.levin_quadrature._adaptive_levin``, through which
``adaptive_levin_sincos`` calls the driver, is wrapped. The wrapper returns the driver's own result
unchanged, and records for each call:
- the test that made it, its ``x_span`` and whether the basis supplied rates;
- every located point (kind, x, theta' there, lambda for a minimum, its window and pieces), every
  minimum found but not located (its lambda), ``num_window_pieces``, ``num_guarded_regions`` and
  ``num_guard_at_depth_max``;
- for a call in which nothing was located and nothing guarded, whether a second call with
  ``_test_only_bypass_location=True`` gives the same value, abserr and region list bit for bit;
- for a call that located a point, the parent's result on the same arguments: the driver of the
  parent ``ad63e8a``, 06c's code (``git show ad63e8a:AdaptiveLevin/levin_quadrature.py``, loaded as a
  module of its own), called with the same basis object, so the change in value can be set
  against both abserrs.

Writes ``out/lg_location.json`` and prints a summary, then the test outcome.
"""

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))

import types  # noqa: E402
import subprocess  # noqa: E402

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402

PARENT_SHA = "ad63e8a"
PARENT_SOURCE = subprocess.run(
    ["git", "show", f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"],
    cwd=MAIN,
    capture_output=True,
    text=True,
    check=True,
).stdout
parent = types.ModuleType("levin_quadrature_parent")
exec(compile(PARENT_SOURCE, "levin_quadrature_parent.py", "exec"), parent.__dict__)

ORIGINAL = lq._adaptive_levin
CALLS = []
CURRENT = {"test": None}


def region_key(region):
    return (
        region.start.hex() if isinstance(region.start, float) else repr(region.start),
        region.end.hex() if isinstance(region.end, float) else repr(region.end),
        region.depth,
        region.type,
        float(region.abserr).hex() if region.abserr is not None else None,
    )


def wrapped(x_span, f, BasisData, *args, **kwargs):
    result = ORIGINAL(x_span, f, BasisData, *args, **kwargs)
    rates = getattr(BasisData, "independent_phase_rates", None)
    supplied = callable(rates) and rates() is not None
    record = {
        "test": CURRENT["test"],
        "x_span": [float(x_span[0]), float(x_span[1])],
        "rates_supplied": supplied,
        "value": float(result["value"]).hex(),
        "abserr": float(result["abserr"]).hex(),
        "stationary_points": [
            {
                key: p[key]
                for key in (
                    "x",
                    "kind",
                    "found_as",
                    "phase",
                    "theta_deriv",
                    "lambda",
                    "window",
                    "window_pieces",
                )
                if key in p
            }
            for p in result["stationary_points"]
        ],
        "unlocated_minima_lambda": [m["lambda"] for m in result["unlocated_minima"]],
        "num_window_pieces": result["num_window_pieces"],
        "num_guarded_regions": result["num_guarded_regions"],
        "num_guard_at_depth_max": result["num_guard_at_depth_max"],
        "num_regions": result["num_regions"],
    }
    if (
        not result["stationary_points"]
        and result["num_guarded_regions"] == 0
        and not kwargs.get("_test_only_bypass_location", False)
    ):
        bypass = ORIGINAL(
            x_span, f, BasisData, *args, _test_only_bypass_location=True, **kwargs
        )
        record["bit_identical_to_bypass"] = (
            float(bypass["value"]).hex() == record["value"]
            and float(bypass["abserr"]).hex() == record["abserr"]
            and [region_key(r) for r in bypass["regions"]]
            == [region_key(r) for r in result["regions"]]
        )
    if result["stationary_points"]:
        before = parent._adaptive_levin(x_span, f, BasisData, *args, **kwargs)
        record["parent"] = {
            "value": float(before["value"]).hex(),
            "abserr": float(before["abserr"]).hex(),
            "num_regions": before["num_regions"],
            "max_depth": before["max_depth"],
            "converged": before["converged"],
        }
        delta = abs(float(result["value"]) - float(before["value"]))
        record["change_in_value"] = delta
        record["change_over_new_abserr"] = (
            delta / result["abserr"] if result["abserr"] > 0 else None
        )
        record["change_over_parent_abserr"] = (
            delta / before["abserr"] if before["abserr"] > 0 else None
        )
        record["new_converged"] = result["converged"]
        record["new_max_depth"] = result["max_depth"]
    CALLS.append(record)
    return result


class Recording(unittest.TextTestResult):
    def startTest(self, test):
        CURRENT["test"] = test.id()
        super().startTest(test)


def main():
    lq._adaptive_levin = wrapped
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(MAIN / "LiouvilleGreen" / "tests"), top_level_dir=str(MAIN)
    )
    runner = unittest.TextTestRunner(
        stream=open("/dev/null", "w"), resultclass=Recording, verbosity=0
    )
    outcome = runner.run(suite)
    lq._adaptive_levin = ORIGINAL

    located = [c for c in CALLS if c["stationary_points"]]
    guarded = [c for c in CALLS if c["num_guarded_regions"] > 0]
    checked = [c for c in CALLS if "bit_identical_to_bypass" in c]
    summary = {
        "tests_run": outcome.testsRun,
        "failures": [t.id() for t, _ in outcome.failures],
        "errors": [t.id() for t, _ in outcome.errors],
        "skipped": len(outcome.skipped),
        "was_successful": outcome.wasSuccessful(),
        "levin_calls": len(CALLS),
        "calls_with_rates_supplied": sum(1 for c in CALLS if c["rates_supplied"]),
        "calls_locating_a_point": len(located),
        "points_by_kind": dict(
            Counter(p["kind"] for c in located for p in c["stationary_points"])
        ),
        "tests_with_a_located_point": sorted({c["test"] for c in located}),
        "calls_guarded": len(guarded),
        "calls_guard_at_depth_max": sum(
            1 for c in CALLS if c["num_guard_at_depth_max"] > 0
        ),
        "calls_with_unlocated_minima": sum(
            1 for c in CALLS if c["unlocated_minima_lambda"]
        ),
        "smallest_unlocated_lambda": min(
            (lam for c in CALLS for lam in c["unlocated_minima_lambda"]),
            default=None,
        ),
        "calls_checked_against_bypass": len(checked),
        "calls_bit_identical_to_bypass": sum(
            1 for c in checked if c["bit_identical_to_bypass"]
        ),
        "located_calls_change_over_new_abserr": [
            c["change_over_new_abserr"] for c in located
        ],
        "located_calls_change_over_parent_abserr": [
            c["change_over_parent_abserr"] for c in located
        ],
        "located_calls_converged_new_parent": [
            [c["new_converged"], c["parent"]["converged"]] for c in located
        ],
        "parent_sha": PARENT_SHA,
        "coalescence_lambda": lq._LEVIN_COALESCENCE_LAMBDA,
        "window_half_phase": lq._LEVIN_WINDOW_HALF_PHASE,
    }
    out = {
        "header": header(Path(__file__), "07: the location on LiouvilleGreen/tests"),
        "summary": summary,
        "located_calls": located,
        "guarded_calls": guarded,
        "calls": CALLS,
    }
    path = OUT / "lg_location.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
