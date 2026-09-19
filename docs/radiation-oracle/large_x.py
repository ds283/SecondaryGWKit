"""The pipeline's `total` against Kohri & Terada's eq. (22) at large x.

Runs from the repository root with no arguments, no Ray and no datastore:

    PYTHONPATH=. ./venv/bin/python docs/radiation-oracle/large_x.py

Its stdout is the three tables of section 8 of `KOHRI-TERADA-ORACLE.md` (about nine seconds on
an Apple M1 Pro).

The fixture of `ComputeTargets/tests/test_quadsource_integral.py` stops at x_resp = 980,
because its Liouville-Green stand-ins do. This script runs the same `evaluate_QuadSource_integral`,
EXACT flavour, at the reference pair (1e-45, 1e-12), at x_resp up to 1e8, by two changes made
from outside the test file, which is not modified:

  * k, q and r are all multiplied by lam = x_resp / 980. u = q/k and v = r/k are unchanged, so
    the Kohri-Terada integral I(v, u, x) being computed is the same function; only the redshift
    at which the response time falls moves, and scaling keeps it where the x_resp = 980 case
    has it. (Without the scaling, x_resp = 1e4 already lands at z < 0 on this background.)
  * `Case._fixtures` is seeded with `Fixture` objects whose Liouville-Green region reaches the
    response redshift (`x_max`), because `Case.__init__` asserts that it does.

In the exact flavour the transfer functions and the Green's function are closed-form stand-ins
(`ExactTkFunctions`, `ExactGk`) valid at every z, so this tests the partition and the Levin
integrator, not the representation floors of the realistic flavour.

Each case is scored against eq. (22) and the head both evaluated at 50 digits
(`eq22_rounding.I_RD_mp`, `head_mp`), so eq. (22)'s own double-precision rounding at small u
(section 7.2) does not enter.

Two smaller tables follow: plain `scipy.quad` of eq. (15) (`kohri_terada.I_RD_quadrature`) as x
grows, for comparison with the pipeline's cost; and `LiouvilleGreen.WKBtools.wrap_theta` at large
theta, which reduces in one step through `WKB_mod_2pi` and is exact and constant-time.
"""

import math
import os
import sys
import time
import warnings
from decimal import Decimal, localcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eq22_rounding import I_RD_mp, head_mp  # noqa: E402
from ComputeTargets.tests import kohri_terada as KT  # noqa: E402
from ComputeTargets.tests.test_quadsource_integral import (  # noqa: E402
    Case,
    SHAPES,
    Shape,
    w_of_b,
)
from ComputeTargets.tests.test_tk_source_functions import Fixture  # noqa: E402
from LiouvilleGreen.WKBtools import wrap_theta  # noqa: E402
from LiouvilleGreen.constants import TWO_PI  # noqa: E402

REF_ATOL, REF_RTOL = 1e-45, 1e-12

# x_resp per shape; the top of each list is the largest x the fixture's Liouville-Green region
# reaches (section 8). Set-up is free at every one of them: the phase is reduced in one step
# by WKB_mod_2pi, so its cost does not grow with x
RUNS = {
    "together": (980.0, 1e4, 1e5, 1e6, 1e7, 1e8),
    "T-first": (980.0, 1e4, 1e5, 1e6, 1e7),
    "q-smooth": (980.0, 1e4, 1e5, 1e6, 1e7),
}


def eq25_without_cos(v, u, x):
    """KT eq. (25) with its cos x term dropped: the large-x limit when |v - u| > sqrt(3)."""
    d = u * u + v * v - 3.0
    log_arg = math.log(abs((3.0 - (u + v) ** 2) / (3.0 - (u - v) ** 2)))
    return (
        3.0 * d / (4.0 * u**3 * v**3 * x) * math.sin(x) * (-4.0 * u * v + d * log_arg)
    )


def pipeline_table():
    w = w_of_b(0.0)

    print("**Table 8.1 -- the pipeline against eq. (22).**")
    print()
    print(
        "| shape | x_resp | lam | x = k tau | z_resp | N + 9/8 | pipeline's declared error "
        "| N + 9/8, head omitted | Levin / abs(total) | eq. (22) vs eq. (25) "
        "| integral | fixture set-up |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for shape_name, x_resps in RUNS.items():
        base = [s for s in SHAPES if s.name == shape_name][0]
        for x_resp in x_resps:
            lam = x_resp / 980.0
            shape = Shape(
                base.name,
                k=lam * base.k,
                q=lam * base.q,
                r=lam * base.r,
                G_cross_x=base.G_cross_x,
            )

            t0 = time.perf_counter()
            for kk in (shape.q, shape.r):
                # the default x_max is 1e3; extend it only where the response time needs it
                Case._fixtures[(w, kk)] = Fixture(
                    w, k=kk, x_max=max(1.0e3, 1.05 * x_resp * kk / shape.r)
                )
            case = Case(b=0.0, shape=shape, x_resp=x_resp, exact=True)
            t1 = time.perf_counter()
            out = case.run(atol=REF_ATOL, rtol=REF_RTOL)
            t2 = time.perf_counter()

            total = float(out["total"])
            halves = abs(float(out["numeric_quad"])) + abs(float(out["WKB_Levin"]))
            pipeline_err = max(float(out["total_abserr"]), REF_RTOL * halves) / abs(
                total
            )

            k = shape.k
            u, v = shape.q / k, shape.r / k
            tau = case.model.functions.tau
            x = k * tau(case.z_resp)
            x_min = k * tau(case.z_source_max)

            I_exact = I_RD_mp(v, u, x)
            head = head_mp(v, u, x, x_min)
            predicted = KT.KT_NORM / (k * k) * float(I_exact - head)
            predicted_no_head = KT.KT_NORM / (k * k) * float(I_exact)
            N = KT.KT_NORM * total / predicted
            N_no_head = KT.KT_NORM * total / predicted_no_head

            asym = (
                eq25_without_cos(v, u, x)
                if abs(v - u) > KT.SQRT3
                else KT.I_RD_asymptotic(v, u, x)
            )
            vs_eq25 = abs(float(I_exact) - asym) / abs(float(I_exact))
            eq25_note = " (no cos term)" if abs(v - u) > KT.SQRT3 else ""

            print(
                f"| {shape_name} | {x_resp:g} | {lam:.3g} | {x:.4e} | {case.z_resp:.3g} "
                f"| {N - KT.KT_NORM:+.2e} | {pipeline_err:.1e} | {N_no_head - KT.KT_NORM:+.2e} "
                f"| {abs(float(out['WKB_Levin'])) / abs(total):.2f} "
                f"| {vs_eq25:.1e}{eq25_note} | {t2 - t1:.2f} s | {t1 - t0:.1f} s |",
                flush=True,
            )


def quadrature_table():
    """Plain scipy.quad of eq. (15) with a breakpoint at every period, at the together shape."""
    import scipy
    from ComputeTargets.tests import kohri_terada as module

    v, u = 1.2e4 / 1.1e4, 1.0e4 / 1.1e4
    calls = [0]
    f_RD = module.f_RD

    def counted(*args):
        calls[0] += 1
        return f_RD(*args)

    print(
        "**Table 8.2 -- plain `scipy.quad` of eq. (15), together shape "
        f"(scipy {scipy.__version__}).**"
    )
    print()
    print(
        "| x | periods of the fastest oscillation | source evaluations | time "
        "| error against eq. (22) | quad's declared error / abs(I) |"
    )
    print("|---|---|---|---|---|---|")
    module.f_RD = counted
    try:
        for x in (1e2, 1e3, 1e4, 3e4, 1e5):
            calls[0] = 0
            t = time.perf_counter()
            value, err = module.I_RD_quadrature(v, u, x)
            dt = time.perf_counter() - t
            exact = float(I_RD_mp(v, u, x))
            periods = x * max(1.0, (u + v) / KT.SQRT3) / (2.0 * math.pi)
            print(
                f"| {x:.0e} | {periods:.0f} | {calls[0]} | {dt:.2f} s "
                f"| {abs(value - exact) / abs(exact):.1e} | {err / abs(exact):.1e} |"
            )
    finally:
        module.f_RD = f_RD


def wrap_theta_table():
    """wrap_theta's cost and rounding at large theta, against reconstruction in a double."""
    print("**Table 8.3 -- `wrap_theta` at large theta.**")
    print()
    print(
        "| theta | time per call | error of wrap_theta | error of theta - div * 2pi "
        "| theta * eps |"
    )
    print("|---|---|---|---|---|")
    for theta in (1e3 + 0.123, 1e5 + 0.123, 1e7 + 0.123):
        # the call is now microseconds rather than the tens of milliseconds the per-cycle loop
        # cost, so a single perf_counter around it is mostly timer noise: average over 1000
        div, mod = wrap_theta(theta)
        t = time.perf_counter()
        for _ in range(1000):
            wrap_theta(theta)
        dt = (time.perf_counter() - t) / 1000.0
        # the exact reduction of the double theta by the double TWO_PI, in 60-digit decimal.
        # wrap_theta reduces in one step through WKB_mod_2pi, whose remainder is an fmod and is
        # exact, so its column is 0 at every theta. The next column is what you get instead by
        # rebuilding the remainder as the double expression theta - div * TWO_PI: that rounds
        # twice, and is the reason the (div, mod) pair is carried rather than reassembled
        with localcontext() as ctx:
            ctx.prec = 60
            exact = Decimal(theta) - div * Decimal(TWO_PI)
            wrap_err = abs(Decimal(mod) - exact)
            one_step_err = abs(Decimal(theta - div * TWO_PI) - exact)
        print(
            f"| {theta:.3e} | {dt * 1e3:.4f} ms | {float(wrap_err):.1e} "
            f"| {float(one_step_err):.1e} | {theta * sys.float_info.epsilon:.1e} |"
        )


def main():
    warnings.simplefilter("ignore")
    print("<!-- generated by docs/radiation-oracle/large_x.py -->")
    print()
    pipeline_table()
    print()
    quadrature_table()
    print()
    wrap_theta_table()


if __name__ == "__main__":
    main()
