"""
Tier A sweeps: accuracy-cost Pareto fronts, the tolerance regime map, the
Chebyshev-order scan, and the phase/core error separation.
"""

import os
import sys
import warnings

import numpy as np

REPO = "/Users/ds283/Documents/Code/SecondaryGWKit"
for _p in (REPO, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mpmath as mp
from math import fmod, pi

from AdaptiveLevin import adaptive_levin_sincos
from LiouvilleGreen.range_reduce_mod_2pi import range_reduce_mod_2pi

from levin_bench.campaign import _write
from levin_bench.problems import PROBLEM_CLASSES, PROBLEMS_BY_NAME, select_omega
from levin_bench.runners import run_levin, run_qawo, run_quad

RESULTS = "results"
mp.mp.dps = 60
TWO_PI_MP = 2 * mp.pi


# ---------------------------------------------------------------------------
# Step 4: accuracy-cost Pareto
# ---------------------------------------------------------------------------

PARETO_TARGETS = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
PARETO_DECADES = [4, 8, 12]


def pareto():
    rows = []
    for cls in PROBLEM_CLASSES:
        for d in PARETO_DECADES:
            w = select_omega(cls, d)
            p = cls(w)
            for target in PARETO_TARGETS:
                atol = target * abs(p.reference)
                for fn, kw in (
                    (run_levin, dict(atol=atol, rtol=target, chebyshev_order=12)),
                    (run_quad, dict(epsabs=atol, epsrel=target, limit=50000)),
                    (run_qawo, dict(epsabs=atol, epsrel=target, limit=50000, maxp1=200)),
                ):
                    r = fn(p, budget_seconds=30.0, **kw)
                    if r is None:
                        continue
                    r["target_rel"] = target
                    r["experiment"] = "A4_pareto"
                    rows.append(r)
            print(f"pareto {p.name} w={w:.4g} done", flush=True)
    return _write(rows, os.path.join(RESULTS, "tierA_pareto.csv"))


# ---------------------------------------------------------------------------
# Step 5: tolerance regime map
# ---------------------------------------------------------------------------
#
# The question is what happens when the caller asks for an absolute tolerance
# the problem cannot deliver.  The adaptive loop responds by bisecting, and once
# a subregion's phase extent drops below 6*pi the driver abandons Levin and
# calls direct quadrature on that subregion -- which at high frequency is
# exactly the thing Levin exists to avoid.

TOL_DECADES = [6, 9, 12]
TOL_ATOLS = [10.0**-e for e in range(6, 31, 2)]


def tolerance_map():
    rows = []
    for name in ("damped_sine", "sinc", "near_singular"):
        cls = PROBLEMS_BY_NAME[name]
        for d in TOL_DECADES:
            w = select_omega(cls, d)
            p = cls(w)
            for atol in TOL_ATOLS:
                r = run_levin(
                    p,
                    atol=atol,
                    rtol=1e-10,
                    chebyshev_order=12,
                    budget_seconds=60.0,
                )
                r["experiment"] = "A5_tolerance"
                r["atol_over_I"] = atol / abs(p.reference)
                rows.append(r)
            print(f"tolerance {name} w={w:.4g} |I|={abs(p.reference):.3g} done", flush=True)
    return _write(rows, os.path.join(RESULTS, "tierA_tolerance.csv"))


# ---------------------------------------------------------------------------
# Step 6: Chebyshev order economics
# ---------------------------------------------------------------------------

ORDERS = [4, 6, 8, 10, 12, 16, 20, 24, 32]
ORDER_DECADES = [6, 12]


def order_scan():
    rows = []
    for cls in PROBLEM_CLASSES:
        for d in ORDER_DECADES:
            w = select_omega(cls, d)
            p = cls(w)
            for order in ORDERS:
                r = run_levin(
                    p,
                    atol=1e-10 * abs(p.reference),
                    rtol=1e-10,
                    chebyshev_order=order,
                    budget_seconds=60.0,
                )
                r["experiment"] = "A6_order"
                rows.append(r)
            print(f"order {p.name} w={w:.4g} done", flush=True)
    return _write(rows, os.path.join(RESULTS, "tierA_order.csv"))


# ---------------------------------------------------------------------------
# Step 8: phase evaluation floor vs Levin core error
# ---------------------------------------------------------------------------
#
# Four ways of supplying the phase to the same integrand:
#
#   exact   theta mod 2pi reduced at 60 digits from the exact double abscissa,
#           so that the phase carries no double-precision reduction error;
#           whatever error remains is the Levin core plus the arithmetic
#   reduce  the repository's prime-factor range reduction
#   fmod    plain fmod(theta(x), 2*pi)
#   naive   no theta_mod_2pi supplied at all
#
# and two integration spans: one whose endpoints make omega*x exactly
# representable, one generic.  The comparison isolates how much of the
# high-frequency error ceiling is phase evaluation and how much is the core.

PHASE_SPANS = {"exact_endpoints": (0.0, 1.0), "generic_endpoints": (0.07, 0.93)}
PHASE_MODES = ("exact", "reduce", "fmod", "naive")


def _damped_sine_ref(w, a, b):
    W, A, B = mp.mpf(w), mp.mpf(a), mp.mpf(b)
    F = lambda X: mp.e ** (-X) * (-mp.sin(W * X) - W * mp.cos(W * X)) / (1 + W * W)
    return float(F(B) - F(A))


def phase_floor():
    from math import exp

    rows = []
    for span_label, (a, b) in PHASE_SPANS.items():
        for d in range(3, 13):
            w = 10.0**d
            ref = _damped_sine_ref(w, a, b)
            for mode in PHASE_MODES:
                th = {"theta": lambda x, w=w: w * x}
                if mode == "exact":
                    th["theta_mod_2pi"] = lambda x, w=w: float(
                        mp.fmod(mp.mpf(w) * mp.mpf(x), TWO_PI_MP)
                    )
                    th["theta_deriv"] = lambda x, w=w: w
                elif mode == "reduce":
                    th["theta_mod_2pi"] = lambda x, w=w: range_reduce_mod_2pi(w, x)[1]
                    th["theta_deriv"] = lambda x, w=w: w
                elif mode == "fmod":
                    th["theta_mod_2pi"] = lambda x, w=w: fmod(w * x, 2 * pi)
                    th["theta_deriv"] = lambda x, w=w: w

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dat = adaptive_levin_sincos(
                        (a, b),
                        [lambda x: exp(-x), lambda x: 0.0],
                        theta=th,
                        atol=1e-10 * abs(ref),
                        rtol=1e-10,
                        chebyshev_order=12,
                        notify_interval=10**9,
                    )
                abs_err = abs(dat["value"] - ref)
                rows.append(
                    {
                        "experiment": "A8_phase_floor",
                        "span": span_label,
                        "a": a,
                        "b": b,
                        "omega": w,
                        "phase_mode": mode,
                        "reference": ref,
                        "abs_reference": abs(ref),
                        "value": dat["value"],
                        "abs_err": abs_err,
                        "rel_err": abs_err / abs(ref),
                        "num_regions": dat["num_regions"],
                        "num_simple_regions": dat["num_simple_regions"],
                        "levin_solves": dat["evaluations"],
                        "max_depth": dat["max_depth"],
                    }
                )
        print(f"phase_floor {span_label} done", flush=True)
    return _write(rows, os.path.join(RESULTS, "tierA_phase_floor.csv"))


# Cost of the range reduction itself, per call
def reduction_cost():
    import timeit

    rows = []
    for d in range(1, 13):
        w = 10.0**d
        x = 0.6180339887498949
        t_red = min(
            timeit.repeat(lambda: range_reduce_mod_2pi(w, x), number=200, repeat=5)
        ) / 200.0
        t_fmod = min(timeit.repeat(lambda: fmod(w * x, 2 * pi), number=200, repeat=5)) / 200.0
        rows.append(
            {
                "experiment": "A8_reduction_cost",
                "omega": w,
                "range_reduce_s": t_red,
                "fmod_s": t_fmod,
                "ratio": t_red / t_fmod,
            }
        )
    return _write(rows, os.path.join(RESULTS, "tierA_reduction_cost.csv"))


# ---------------------------------------------------------------------------
# Estimator fidelity: does the integrator's own error estimate predict the
# true error?
# ---------------------------------------------------------------------------
#
# The driver accepts a region when |estimate - refined_estimate| < atol or the
# relative version of the same quantity < rtol, where refined_estimate is the
# sum over the region's two halves.  Both values are built from the SAME
# endpoint phases, so an error in theta cancels out of the difference and is
# invisible to the test.  This experiment aggregates the per-region estimates
# into the total error estimate the driver does not currently return, and
# compares it against the true error measured from the closed-form oracle.

FIDELITY_DECADES = list(range(1, 13))


def estimator_fidelity():
    rows = []
    for cls in PROBLEM_CLASSES:
        for d in FIDELITY_DECADES:
            w = select_omega(cls, d)
            p = cls(w)
            r = run_levin(
                p,
                atol=1e-10 * abs(p.reference),
                rtol=1e-10,
                chebyshev_order=12,
                budget_seconds=60.0,
                keep_regions=True,
            )
            regions = r.pop("_regions", None) or []
            est_abs = float(sum(abs(reg.abserr) for reg in regions if reg.abserr is not None))
            r["experiment"] = "A9_estimator"
            r["internal_abs_est"] = est_abs
            r["internal_rel_est"] = est_abs / abs(p.reference)
            r["true_abs_err"] = r["abs_err"]
            r["fidelity_ratio"] = r["abs_err"] / est_abs if est_abs > 0 else np.nan
            rows.append(r)
        print(f"fidelity {p.name} done", flush=True)
    return _write(rows, os.path.join(RESULTS, "tierA_estimator.csv"))


if __name__ == "__main__":
    fns = {
        "fidelity": estimator_fidelity,
        "pareto": pareto,
        "tolerance": tolerance_map,
        "order": order_scan,
        "phase": phase_floor,
        "redcost": reduction_cost,
    }
    for which in sys.argv[1:]:
        print("wrote", fns[which]())
