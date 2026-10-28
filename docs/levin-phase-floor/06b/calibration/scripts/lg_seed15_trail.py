"""levin-phase-floor prompt 06b, section 2 test 5: the one seeded LiouvilleGreen call that reaches
depth_max under the new code.

lg_seeded_draws.py finds one Levin call, of 4,320, whose maximum depth moves to 20 under 06b's
code: test_JJJ at seed 15, its second draw (J110), phase group 3. This script rebuilds that call
exactly as the test makes it -- ``random.seed(15)``, the test's draws in order, the phase objects
built as ``compute_3Bessel`` builds them, ``quad_JJJ`` at the test's tolerances -- records the
group-3 call at the parent (the bypass) and at the new code, and then follows the bisection
ancestry of three points of the region the parent accepted through the round-off-limited exit. At
each depth it gives the region's step-(4) residual (or, on the Clenshaw-Curtis fallback, its
nested-pair estimate), its relative error against the driver's denominator, its eq. (151) floor,
the length-scaled atol, and the ratio of its halves' floors to its own (the premise; 06b's rho is
1/2).

Writes ../out/lg_seed15_trail.txt. No Ray and no datastore.
"""

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

REPO = prov.REPO
sys.path.insert(0, str(REPO))

import AdaptiveLevin.levin_quadrature as lq  # noqa: E402
from LiouvilleGreen.bessel_phase import (  # noqa: E402
    DEFAULT_AMPLITUDE_RTOL,
    DEFAULT_PHASE_ATOL,
    bessel_phase,
)
from LiouvilleGreen.three_bessel_integrals import (  # noqa: E402
    DEFAULT_3BESSEL_CHEBYSHEV_ORDER,
    quad_JJJ,
)
from LiouvilleGreen.tests.test_3bessel_analytic import (  # noqa: E402
    MAX_X,
    Jintegrals,
)

OUT = prov.OUT / "lg_seed15_trail.txt"
SEED = 15
DRAW = 1  # the second oracle of test_JJJ, J110
GROUP_CALL = 2  # the third of the four phase-group calls
QUAD_ATOL, QUAD_RTOL = 1e-14, 1e-10  # compute_3Bessel's defaults, which test_JJJ uses
TARGETS = (10.6797, 10.6850, 12.0)


def draws():
    random.seed(SEED)
    out = []
    for _ in Jintegrals:
        out.append(
            (
                random.uniform(0.1, 5.0),
                random.uniform(0.1, 5.0),
                random.uniform(0.1, 5.0),
            )
        )
    return out


def capture(J, k, q, s):
    original = lq._adaptive_levin
    calls = {}
    for mode in ("parent", "new"):
        store = []

        def wrapper(*args, **kwargs):
            if mode == "parent":
                kwargs["_test_only_bypass_roundoff_premise"] = True
            data = original(*args, **kwargs)
            store.append((args, kwargs, data))
            return data

        lq._adaptive_levin = wrapper
        try:
            phases = [
                bessel_phase(
                    order + 0.5,
                    1.075 * c * MAX_X,
                    phase_atol=DEFAULT_PHASE_ATOL,
                    amplitude_rtol=DEFAULT_AMPLITUDE_RTOL,
                )
                for order, c in ((J.mu, k), (J.nu, q), (J.sigma, s))
            ]
            quad_JJJ(
                *phases,
                J.mu,
                J.nu,
                J.sigma,
                k,
                q,
                s,
                MAX_X,
                atol=QUAD_ATOL,
                rtol=QUAD_RTOL,
            )
        finally:
            lq._adaptive_levin = original
        calls[mode] = store[GROUP_CALL]
    return calls


def trail(lines, args, kwargs, target):
    (a0, b0), f, basis = args[0], args[1], args[2]
    order = kwargs["chebyshev_order"]
    atol, rtol = kwargs["atol"], kwargs["rtol"]
    width = abs(b0 - a0)
    lo, hi = a0, b0
    vectorize_cache = {}

    def solve(span):
        return lq._adaptive_levin_subregion(
            span,
            f,
            basis,
            None,
            chebyshev_order=order,
            rtol=rtol,
            vectorize_cache=vectorize_cache,
        )

    lines.append(f"-- the ancestry of log x = {target}")
    for depth in range(lq.DEFAULT_LEVIN_MAX_DEPTH + 1):
        c = (hi + lo) / 2.0
        local_atol = lq._local_atol(atol, lo, hi, width)
        data = solve((lo, hi))
        if data["is_direct"]:
            abserr = data["abserr_direct"]
            denom = max(abs(data["value"]), local_atol)
            kind, premise = "CC   ", ""
        else:
            left, right = solve((lo, c)), solve((c, hi))
            refined = left["value"] + right["value"]
            abserr = abs(data["value"] - refined)
            denom = max(min(abs(data["value"]), abs(refined)), local_atol)
            kind = "Levin"
            premise = (
                f" halves/own floor {(left['phase_err'] + right['phase_err']) / data['phase_err']:.3g}"
                if data["phase_err"] > 0
                else ""
            )
        relerr = abserr / denom
        lines.append(
            f"   depth {depth:2d} [{lo:.6f}, {hi:.6f}] {kind} value {data['value']:+.4e}"
            f" residual {abserr:.3e} relerr {relerr:.3e} floor {data['phase_err']:.3e}"
            f" local_atol {local_atol:.2e}{premise}"
        )
        if abserr < local_atol or relerr < rtol:
            lines.append("   resolved here")
            return
        if lo <= target < c:
            hi = c
        else:
            lo = c


def main():
    lines = prov.header(
        __file__,
        "section 2 test 5, the seed-15 call that reaches depth_max under the new code",
        (
            REPO / "LiouvilleGreen" / "tests" / "test_3bessel_analytic.py",
            REPO / "LiouvilleGreen" / "three_bessel_integrals.py",
            REPO / "LiouvilleGreen" / "bessel_phase.py",
        ),
    )
    triples = draws()
    J = Jintegrals[DRAW]
    k, q, s = triples[DRAW]
    lines.append(
        f"# test_JJJ seed {SEED}, draw {DRAW + 1} ({J.__name__}): k={k!r}, q={q!r}, s={s!r};"
        f" quad atol {QUAD_ATOL}, rtol {QUAD_RTOL}; Chebyshev order"
        f" {DEFAULT_3BESSEL_CHEBYSHEV_ORDER}"
    )
    lines.append("")

    calls = capture(J, k, q, s)
    for mode in ("parent", "new"):
        args, kwargs, data = calls[mode]
        lines.append(
            f"== {mode}: {kwargs.get('notify_label')} x_span {args[0]!r}: value"
            f" {data['value']!r} abserr {data['abserr']!r} regions {data['num_regions']}"
            f" phase-limited {data['num_phase_limited_regions']} max depth {data['max_depth']}"
            f" solves {data['evaluations']} converged {data['converged']}"
        )
        by_depth = {}
        for region in data["regions"]:
            key = (region.depth, region.type)
            by_depth[key] = by_depth.get(key, 0) + 1
        lines.append(f"   regions by (depth, type): {sorted(by_depth.items())}")
        lines.append(
            f"   abserr by source: resolution {data['abserr_resolution']:.4e} roundoff"
            f" {data['abserr_roundoff']:.4e} fallback {data['abserr_fallback']:.4e}"
        )
        deep = [r for r in data["regions"] if r.depth >= 18]
        if deep:
            lines.append(
                f"   regions at depth >= 18: {len(deep)}, spanning [{deep[0].start!r},"
                f" {deep[-1].end!r}]"
            )
        if mode == "parent" or data["num_regions"] <= 12:
            for region in data["regions"]:
                lines.append(f"   region {region}")
        else:
            worst = sorted(data["regions"], key=lambda r: -r.total_err)[:3]
            for region in worst:
                lines.append(f"   largest-error region {region}")
    lines.append("")

    args, kwargs, _ = calls["new"]
    for target in TARGETS:
        trail(lines, args, kwargs, target)
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
