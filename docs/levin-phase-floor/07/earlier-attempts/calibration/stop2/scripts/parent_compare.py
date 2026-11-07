"""levin-phase-floor 07: the new driver against the parent's (06b's code), on the problems of
``AdaptiveLevin/tests/test_levin_quadrature.py``, each with and without a closed-form
``theta_deriv`` where one exists, and prompt 06b's two premise problems.

The parent is 06b's code, ``git show cf3efac:AdaptiveLevin/levin_quadrature.py``, loaded as a module of its
own. Each problem is run through both ``adaptive_levin_sincos`` (the generic m = 4 problem through
both ``_adaptive_levin``, with the test module's ``_TwoIndependentSinCosBasis``), and the value,
abserr and region list (start, end, depth, type, abserr, by ``float.hex``) are compared. For the new
driver the location's diagnostics are recorded too.

Writes ``out/parent_compare.json``.
"""

import json
import subprocess
import sys
import types
from math import atan, exp, pi
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from prov import MAIN, OUT, header  # noqa: E402

sys.path.insert(0, str(MAIN))
import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
from AdaptiveLevin.tests.test_levin_quadrature import (  # noqa: E402
    _TwoIndependentSinCosBasis,
)

PARENT_SHA = "cf3efac"


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


def mod2pi(w):
    def theta_mod_2pi(x):
        v = np.fmod(w * x, 2.0 * pi)
        return v + 2.0 * pi if v < 0.0 else v

    return theta_mod_2pi


def problems():
    one = [lambda x: 1.0, lambda x: 0.0]
    damped = [lambda x: exp(-x), lambda x: 0.0]
    out = {
        "sin": ((1.0, 5.0e4), one, {"theta": lambda x: x}, (1e-15, 1e-10, 12)),
        "sin+deriv": (
            (1.0, 5.0e4),
            one,
            {"theta": lambda x: x, "theta_deriv": lambda x: 1.0},
            (1e-15, 1e-10, 12),
        ),
        "cos": (
            (1.0, 5.0e5),
            [lambda x: 0.0, lambda x: 1.0],
            {"theta": lambda x: x},
            (1e-15, 1e-10, 12),
        ),
        "sinc": (
            (1.0, 100.0),
            [lambda x: 1.0 / x, lambda x: 0.0],
            {"theta": lambda x: 100.0 * x},
            (1e-15, 1e-10, 12),
        ),
        "sinc+deriv": (
            (1.0, 100.0),
            [lambda x: 1.0 / x, lambda x: 0.0],
            {"theta": lambda x: 100.0 * x, "theta_deriv": lambda x: 100.0},
            (1e-15, 1e-10, 12),
        ),
        "lstsq": (
            (0.0, 1.0),
            [lambda x: exp(-x), lambda x: 0.3],
            {"theta": lambda x: 10.0 * pi * x, "theta_deriv": lambda x: 10.0 * pi},
            (1e-15, 1e-12, 16),
        ),
        "reduced": (
            (1.0 / 3.0, 7.0 / 3.0),
            damped,
            {
                "theta": lambda x: 1e6 * x,
                "theta_mod_2pi": mod2pi(1e6),
                "theta_deriv": lambda x: 1e6,
            },
            (1e-18, 1e-13, 12),
        ),
        "mod2pi-only": (
            (1.0 / 3.0, 7.0 / 3.0),
            damped,
            {"theta": lambda x: 1e6 * x, "theta_mod_2pi": mod2pi(1e6)},
            (1e-18, 1e-13, 12),
        ),
        "reversed+deriv": (
            (5.0, 1.0),
            one,
            {"theta": lambda x: 100.0 * x, "theta_deriv": lambda x: 100.0},
            (1e-13, 1e-11, 16),
        ),
        "bump": (
            (0.0, 3.0),
            [lambda x: exp(-400.0 * (x - 1.1) ** 2), lambda x: 0.0],
            {"theta": lambda x: 3.0e4 * x, "theta_deriv": lambda x: 3.0e4},
            (1e-10, 1e-7, 16),
        ),
        "c2-stationary": (
            (0.0, 1.0),
            damped,
            {"theta": lambda x: 1e6 * (x - x * x)},
            (1e-10, 1e-10, 16),
        ),
        "c2-stationary+deriv": (
            (0.0, 1.0),
            damped,
            {
                "theta": lambda x: 1e6 * (x - x * x),
                "theta_deriv": lambda x: 1e6 * (1.0 - 2.0 * x),
            },
            (1e-10, 1e-10, 16),
        ),
    }
    # prompt 06b's two premise problems (AdaptiveLevin/tests/test_levin_roundoff_exit.py tests 1
    # and 3), where the round-off-limited exit's premise check decides acceptance
    out["06b-exp-levin"] = (
        (0.0, 20.0),
        one,
        {
            "theta": lambda x: 100.0 * np.exp(x),
            "theta_deriv": lambda x: 100.0 * np.exp(x),
        },
        (1e-15, 1e-7, 16),
    )
    out["06b-exp-fallback"] = (
        (0.0, 6.0),
        [lambda x: np.exp(x), lambda x: 0.0],
        {
            "theta": lambda x: 1e-6 * np.exp(x),
            "theta_deriv": lambda x: 1e-6 * np.exp(x),
        },
        (1e-20, 1e-10, 16),
    )
    for lam in (10.0, 100.0, 1000.0):
        f = [lambda x: 0.0, lambda x: 1.0 / (1.0 + x * x)]
        out[f"grz{lam:g}"] = (
            (-1.0, 1.0),
            f,
            {"theta": lambda x, lam=lam: lam * atan(x)},
            (1e-15, 1e-10, 12),
        )
        out[f"grz{lam:g}+deriv"] = (
            (-1.0, 1.0),
            f,
            {
                "theta": lambda x, lam=lam: lam * atan(x),
                "theta_deriv": lambda x, lam=lam: lam / (1.0 + x * x),
            },
            (1e-15, 1e-10, 12),
        )
    return out


def region_key(r):
    return (
        float(r.start).hex(),
        float(r.end).hex(),
        r.depth,
        r.type,
        None if r.abserr is None else float(r.abserr).hex(),
    )


def compare(a, b):
    return {
        "value_identical": float(a["value"]).hex() == float(b["value"]).hex(),
        "abserr_identical": float(a["abserr"]).hex() == float(b["abserr"]).hex(),
        "regions_identical": [region_key(r) for r in a["regions"]]
        == [region_key(r) for r in b["regions"]],
        "parent_value": a["value"],
        "new_value": b["value"],
        "parent_abserr": a["abserr"],
        "new_abserr": b["abserr"],
        "parent_regions": a["num_regions"],
        "new_regions": b["num_regions"],
        "located": [(p["kind"], p["x"]) for p in b["stationary_points"]],
        "window_pieces": b["num_window_pieces"],
        "guarded_regions": b["num_guarded_regions"],
        "new_converged": b["converged"],
        "parent_converged": a["converged"],
    }


def main():
    parent = load_parent()
    out = {"header": header(Path(__file__), "07: new driver against the parent")}
    rows = {}
    for name, (span, f, theta, (atol, rtol, order)) in problems().items():
        kw = dict(atol=atol, rtol=rtol, chebyshev_order=order)
        a = parent.adaptive_levin_sincos(span, f, theta, **kw)
        b = lq.adaptive_levin_sincos(span, f, theta, **kw)
        rows[name] = compare(a, b)
    f4 = [lambda x: 1.0, lambda x: 0.0, lambda x: 0.0, lambda x: 1.0]
    basis = _TwoIndependentSinCosBasis(1.0, 2.0)
    kw = dict(atol=1e-13, rtol=1e-11, chebyshev_order=16)
    a = parent._adaptive_levin((1.0, 50.0), f4, basis, **kw)
    b = lq._adaptive_levin((1.0, 50.0), f4, basis, **kw)
    rows["generic-m4"] = compare(a, b)
    out["rows"] = rows
    out["identical"] = sorted(
        n
        for n, r in rows.items()
        if r["value_identical"] and r["abserr_identical"] and r["regions_identical"]
    )
    out["moved"] = sorted(set(rows) - set(out["identical"]))
    for name, r in rows.items():
        print(
            f"{name:22s} identical={r['value_identical'] and r['abserr_identical'] and r['regions_identical']} "
            f"located={r['located']} guarded={r['guarded_regions']} "
            f"parent abserr {r['parent_abserr']:.3g} new {r['new_abserr']:.3g}"
        )
    path = OUT / "parent_compare.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
