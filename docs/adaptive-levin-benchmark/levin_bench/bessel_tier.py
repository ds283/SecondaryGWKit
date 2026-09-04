"""
Tier B: the Bessel application layer.

Drives `quad_JJJ` / `quad_YJJ` against the seven closed forms already present in
the repository's own analytic test module.  The oracle classes are *imported*
rather than transcribed: the review documents a transcription typo in exactly
this family of formulae, and copying them here would just create a second place
for one to live.

Three sub-experiments:

  B0  truncation scan   -- fixed wavenumbers, max_x swept over ten decades, so
                           that truncation of the (conditionally convergent)
                           x -> infinity oracle can be separated from
                           quadrature error.
  B1  production scan   -- max_x = 1e12 as the repository uses, overall
                           wavenumber scale swept over four decades, all seven
                           oracles.  Levin only, with the Liouville-Green phase
                           build timed separately.
  B2  head-to-head      -- Levin against plain scipy.quad on the raw
                           spherical-Bessel triple product, both over the same
                           finite interval, both scored against the closed form
                           at an upper limit where truncation is negligible.

A note on QAWO.  QUADPACK's oscillatory rule integrates amplitude(x) * sin(w x)
for a phase *linear* in x with a *single* frequency.  A product of three Bessel
functions is neither: even after the Liouville-Green sum-and-difference
decomposition the group phase is theta_mu(kx) +- theta_nu(qx) +- theta_sigma(sx),
which is nonlinear in x.  QAWO therefore cannot be applied in this tier at all,
and is not reported for it.
"""

import os
import sys
import time
import warnings

import numpy as np

REPO = "/Users/ds283/Documents/Code/SecondaryGWKit"
for _p in (REPO, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scipy.integrate import quad
from scipy.special import spherical_jn, spherical_yn

from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.tests.test_3bessel_analytic import Jintegrals, Yintegrals
from LiouvilleGreen.three_bessel_integrals import quad_JJJ, quad_YJJ

from levin_bench.campaign import _write
from levin_bench.runners import BudgetExceeded, _Budget

RESULTS = "results"

# base wavenumber triple: satisfies the triangle inequality, no two equal, not a
# right-angle or degenerate configuration
K0, Q0, S0 = 1.3, 1.7, 2.1

PHASE_ATOL = 1e-25
PHASE_RTOL = 5e-14
QUAD_ATOL = 1e-14
QUAD_RTOL = 1e-10

PHASE_HEADROOM = 1.075

ORACLES = [("JJJ", J, quad_JJJ) for J in Jintegrals] + [
    ("YJJ", Y, quad_YJJ) for Y in Yintegrals
]


def _oracle_name(kind, O):
    return f"{kind[0]}{int(O.mu)}{int(O.nu)}{int(O.sigma)}"


def build_phases(O, k, q, s, max_x, sample_points=None):
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph = [
            bessel_phase(
                order + 0.5,
                PHASE_HEADROOM * mom * max_x,
                sample_points=sample_points,
                atol=PHASE_ATOL,
                rtol=PHASE_RTOL,
            )
            for order, mom in ((O.mu, k), (O.nu, q), (O.sigma, s))
        ]
    return ph, time.perf_counter() - t0


def run_bessel_levin(
    kind, O, evaluator, k, q, s, max_x, sample_points=None, chebyshev_order=None
):
    """One Levin evaluation of a three-Bessel integral, timed and scored."""
    ph, phase_time = build_phases(O, k, q, s, max_x, sample_points=sample_points)

    kwargs = {}
    if chebyshev_order is not None:
        kwargs["chebyshev_order"] = chebyshev_order

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # quad_JJJ/quad_YJJ return a BesselIntegralResult (value, abserr, converged,
        # phase_limited) as of prompts/levin-refactor's prompt 09, not a bare float -- see
        # LiouvilleGreen/three_bessel_integrals.py.
        result = evaluator(
            ph[0],
            ph[1],
            ph[2],
            O.mu,
            O.nu,
            O.sigma,
            k,
            q,
            s,
            max_x,
            QUAD_ATOL,
            QUAD_RTOL,
            **kwargs,
        )
    wall = time.perf_counter() - t0

    value = result.value
    ref = O.analytic(k, q, s)
    abs_err = abs(value - ref)
    rel_err = abs_err / abs(ref) if ref != 0.0 else np.nan

    # the harness exists to compare reported against true, so record both: the propagated
    # abserr this evaluator now reports, and whether the measured abs_err above actually falls
    # within it (it does not always -- the reported abserr is a quadrature-only estimate that
    # cannot see the phase/modulus splines' own fit error; see BesselIntegralResult's docstring
    # and prompts/levin-refactor/logs/09-caller-propagation.md)
    reported_abserr = result.abserr
    abserr_ratio = abs_err / reported_abserr if reported_abserr > 0 else np.nan

    return {
        "tier": "B",
        "oracle": _oracle_name(kind, O),
        "kind": kind,
        "method": "levin",
        "mu": O.mu,
        "nu": O.nu,
        "sigma": O.sigma,
        "k": k,
        "q": q,
        "s": s,
        "max_x": max_x,
        "n_osc": (k + q + s) * max_x / (2.0 * np.pi),
        "sample_points": -1 if sample_points is None else sample_points,
        "chebyshev_order": -1 if chebyshev_order is None else chebyshev_order,
        "reference": ref,
        "value": value,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "reported_abserr": reported_abserr,
        "abserr_bounds_truth": bool(abs_err <= reported_abserr),
        "abserr_ratio_true_over_reported": abserr_ratio,
        "converged": result.converged,
        "phase_limited": result.phase_limited,
        "time_s": wall,
        "phase_build_s": phase_time,
        "total_s": wall + phase_time,
        "status": "ok",
    }


def run_bessel_quad(kind, O, k, q, s, max_x, budget_seconds=60.0, panels_per_osc=2):
    """
    Brute-force baseline: scipy.quad on the raw spherical-Bessel triple product
    over [0, max_x], split into panels so that QUADPACK is never asked to
    resolve many oscillations inside one call (without the split it simply
    reports non-convergence and returns garbage).
    """
    if kind == "JJJ":
        integrand = lambda x: (
            x
            * x
            * spherical_jn(int(O.mu), k * x)
            * spherical_jn(int(O.nu), q * x)
            * spherical_jn(int(O.sigma), s * x)
        )
    else:
        integrand = lambda x: (
            x
            * x
            * spherical_yn(int(O.mu), k * x)
            * spherical_jn(int(O.nu), q * x)
            * spherical_jn(int(O.sigma), s * x)
        )

    n_osc = (k + q + s) * max_x / (2.0 * np.pi)
    n_panels = max(1, int(panels_per_osc * n_osc))

    b = _Budget(seconds=budget_seconds)
    counted = b.amp(integrand)
    edges = np.linspace(0.0, max_x, n_panels + 1)

    t0 = time.perf_counter()
    total = 0.0
    status = "ok"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for lo, hi in zip(edges[:-1], edges[1:]):
                total += quad(
                    counted, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=200
                )[0]
    except BudgetExceeded:
        status = "timeout"
        total = np.nan
    wall = time.perf_counter() - t0

    ref = O.analytic(k, q, s)
    abs_err = abs(total - ref) if status == "ok" else np.nan
    rel_err = abs_err / abs(ref) if (status == "ok" and ref != 0.0) else np.nan

    return {
        "tier": "B",
        "oracle": _oracle_name(kind, O),
        "kind": kind,
        "method": "quad",
        "mu": O.mu,
        "nu": O.nu,
        "sigma": O.sigma,
        "k": k,
        "q": q,
        "s": s,
        "max_x": max_x,
        "n_osc": n_osc,
        "n_panels": n_panels,
        "reference": ref,
        "value": total,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "time_s": wall,
        "phase_build_s": 0.0,
        "total_s": wall,
        "n_amp_eval": b.n_amp,
        "status": status,
    }


# ---------------------------------------------------------------------------
# B0: truncation scan
# ---------------------------------------------------------------------------


def truncation_scan():
    rows = []
    for kind, O, ev in ORACLES:
        for d in range(2, 13):
            max_x = 10.0**d
            r = run_bessel_levin(kind, O, ev, K0, Q0, S0, max_x)
            r["experiment"] = "B0_truncation"
            rows.append(r)
            print(
                f"{r['oracle']:>6s} max_x=1e{d:<2d} nosc={r['n_osc']:.3g} "
                f"rel_err={r['rel_err']:.3e} t={r['time_s']:.4g}s "
                f"phase={r['phase_build_s']:.4g}s",
                flush=True,
            )
        print(flush=True)
    return _write(rows, os.path.join(RESULTS, "tierB_truncation.csv"))


# ---------------------------------------------------------------------------
# B1: production scan over overall wavenumber scale
# ---------------------------------------------------------------------------

B1_MAX_X = 1.0e12
# kappa=1000 (x_max ~ 8e15) does not complete a bessel_phase build within ~25 min:
# the phase layer, not the Levin core, is the scalability limit.  Capped at 100.
B1_KAPPA = [1.0, 10.0, 100.0]


def production_scan():
    rows = []
    for kind, O, ev in ORACLES:
        for kappa in B1_KAPPA:
            r = run_bessel_levin(
                kind, O, ev, kappa * K0, kappa * Q0, kappa * S0, B1_MAX_X
            )
            r["experiment"] = "B1_production"
            r["kappa"] = kappa
            rows.append(r)
            print(
                f"{r['oracle']:>6s} kappa={kappa:<8g} nosc={r['n_osc']:.3g} "
                f"rel_err={r['rel_err']:.3e} levin={r['time_s']:.4g}s "
                f"phase={r['phase_build_s']:.4g}s",
                flush=True,
            )
        print(flush=True)
    return _write(rows, os.path.join(RESULTS, "tierB_production.csv"))


# ---------------------------------------------------------------------------
# B2: head-to-head against brute force at a common finite upper limit
# ---------------------------------------------------------------------------

# one low-order and one high-order JJJ case; YJJ is excluded because the
# spherical y_mu factor diverges at the origin and a brute-force sweep from
# x = 0 is not a meaningful baseline there
B2_ORACLES = [
    (kind, O, ev)
    for kind, O, ev in ORACLES
    if kind == "JJJ" and _oracle_name(kind, O) in ("J000", "J222")
]
B2_KAPPA = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def head_to_head(max_x=1.0e6, budget_seconds=60.0):
    """
    Both methods over [0, max_x] with the closed form as oracle.  max_x is
    chosen from the B0 scan to put truncation error below the quadrature error
    of either method.
    """
    rows = []
    for kind, O, ev in B2_ORACLES:
        for kappa in B2_KAPPA:
            k, q, s = kappa * K0, kappa * Q0, kappa * S0

            r = run_bessel_levin(kind, O, ev, k, q, s, max_x)
            r["experiment"] = "B2_head_to_head"
            r["kappa"] = kappa
            rows.append(r)
            print(
                f"{r['oracle']:>6s} kappa={kappa:<6g} nosc={r['n_osc']:.3g} levin "
                f"rel_err={r['rel_err']:.3e} t={r['total_s']:.4g}s "
                f"(phase {r['phase_build_s']:.3g}s)",
                flush=True,
            )

            r = run_bessel_quad(
                kind, O, k, q, s, max_x, budget_seconds=budget_seconds
            )
            r["experiment"] = "B2_head_to_head"
            r["kappa"] = kappa
            rows.append(r)
            print(
                f"{'':>6s} {'':>12s} {'':>16s} quad  "
                f"rel_err={r['rel_err']:.3e} t={r['total_s']:.4g}s "
                f"n={r['n_amp_eval']} status={r['status']}",
                flush=True,
            )
            print(flush=True)
    return _write(rows, os.path.join(RESULTS, "tierB_head_to_head.csv"))


if __name__ == "__main__":
    which = sys.argv[1]
    fns = {
        "truncation": truncation_scan,
        "production": production_scan,
        "head": head_to_head,
    }
    print("wrote", fns[which]())
