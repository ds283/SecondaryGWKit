"""levin-phase-floor 07 section 3 tests 1 and 5: the parent's code on their problems. New in 07.

Test 1 asks that the parent (the dispatch commit's AdaptiveLevin/levin_quadrature.py, 06m's code) return a Levin
leaf across the stationary point, wrong beyond its abserr, and that its value, abserr, region count and
`converged` be recorded in the test's comment; test 5 asks for a case where the parent marks a leaf across the
point floor-limited. This runs the parent, loaded from git show, on test_references.py's P1 (test 1) and P5 (test
5), at the tests' arguments (x_span (0, 16), atol 1e-20, rtol 1e-7, the module's other defaults), and writes what
it returns, against the references in out/run2/test_references.json, to out/run2/parent_cases.json.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import prov  # noqa: E402
import test_references as T  # noqa: E402

sys.path.insert(0, str(prov.REPO))


def main():
    parent, sha = prov.load_module_from_source(
        prov.git("show", f"{prov.PARENT_SHA}:AdaptiveLevin/levin_quadrature.py"),
        "levin_quadrature_parent_bd21bf2",
    )
    refs = json.loads((prov.OUT / "test_references.json").read_text())
    out = {
        "header": prov.json_header(
            __file__,
            "section 3 tests 1 and 5, the parent on their problems",
            (HERE / "test_references.py", prov.OUT / "test_references.json"),
        ),
        "parent_module_sha256": sha,
    }
    for name, p in (("P1", T.P1), ("P5", T.P5)):
        A, x_s, g, h, L = p["A"], p["x_s"], p["g"], p["h"], p["L"]
        a = A * math.exp(x_s)
        theta = {
            "theta": lambda x: A * np.exp(x) - a * x,
            "theta_deriv": lambda x: A * np.exp(x) - a,
        }
        f = [lambda x: np.exp(g * (x - x_s)), lambda x: np.cos(h * x)]
        r = parent.adaptive_levin_sincos((0.0, L), f, theta, atol=1e-20, rtol=1e-7)
        ref = float.fromhex(refs[name]["value_hex"])
        rec = {
            "value_hex": float(r["value"]).hex(),
            "value": r["value"],
            "abserr": r["abserr"],
            "true_error": abs(r["value"] - ref),
            "error_over_abserr": abs(r["value"] - ref) / r["abserr"],
            "num_regions": r["num_regions"],
            "converged": r["converged"],
            "num_phase_limited_regions": r["num_phase_limited_regions"],
            "regions": [str(region) for region in r["regions"]],
            "region_holds_x_s": [
                min(region.start, region.end) < x_s < max(region.start, region.end)
                for region in r["regions"]
            ],
        }
        out[name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if k != "regions"}))
    (prov.OUT / "parent_cases.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
