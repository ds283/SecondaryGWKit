"""levin-phase-floor prompt 06a section 5 item 7: the stationary-point scan, for 07. New in 06a.

For one Levin call (one adaptive_levin_sincos call of a unit at rtol 1e-7), on the call's own
``theta["theta_deriv"]`` (the callable the driver was handed, kept by the worker's wrapper):

- **The scan.** theta' is evaluated, one scalar call per point (as the driver samples a callable
  that does not vectorise), on N = 4,097 points uniform in the call's abscissa (log(1+z')) over its
  ``x_span``, endpoints included (numpy.linspace). Recorded: the number of sign changes (between
  consecutive non-zero samples; an exact zero sample is counted with the change it lies in);
  min|theta'| / max|theta'| over the grid; and each change's abscissa, as the midpoint of its
  bracketing pair, to the grid spacing h = |b - a| / (N - 1), which is recorded.
- **Accepted regions.** For each accepted region (the call's ``regions``), listed if a sign change
  lies between two consecutive grid points that are both inside the region (``how = "grid"``).
  Where the grid shows none inside a region, theta' is evaluated at the region's two ends and its
  midpoint; if those disagree in sign, the region is listed too (``how = "ends"``). For a listed
  region: its depth, branch (Levin or Clenshaw-Curtis), abserr, relerr, phase_err and
  phase_limited. A region's value is not kept by the driver; its relerr is abserr over
  max(min(|estimate|, |refined estimate|), local atol) (Levin) or max(|estimate|, local atol)
  (Clenshaw-Curtis), so 1/relerr is the region's |value| against its abserr where the value is
  above the local atol (``levin_quadrature.py``, ``_adaptive_levin``).
- **Its limits.** A pair of stationary points closer together than h is not seen by the grid.

It reads phase objects only, after the unit's timed section, with the dispatcher uninstalled.
"""

import math
import time

import numpy as np

N_GRID = 4097


def _sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def scan_call(theta_deriv, x_span, regions, n=N_GRID):
    t0 = time.perf_counter()
    a, b = float(x_span[0]), float(x_span[1])
    xs = np.linspace(a, b, n)
    d = np.array([float(theta_deriv(float(x))) for x in xs])
    h = abs(b - a) / (n - 1)
    finite = np.isfinite(d)
    absd = np.abs(d[finite])
    ratio = float(absd.min() / absd.max()) if absd.size and absd.max() > 0 else float("nan")
    # sign changes between consecutive non-zero samples
    changes = []  # (index_left, index_right) of the bracketing non-zero samples
    last = None
    for i in range(n):
        s = _sign(d[i]) if finite[i] else 0
        if s == 0:
            continue
        if last is not None and _sign(d[last]) != s:
            changes.append((last, i))
        last = i
    change_x = [0.5 * (float(xs[i]) + float(xs[j])) for i, j in changes]
    change_halfwidth = [0.5 * abs(float(xs[j]) - float(xs[i])) for i, j in changes]

    listed = []
    n_ends_evals = 0
    for ri, reg in enumerate(regions):
        lo, hi = min(reg.start, reg.end), max(reg.start, reg.end)
        how = None
        for i, j in changes:
            xi, xj = float(xs[i]), float(xs[j])
            if lo <= min(xi, xj) and max(xi, xj) <= hi:
                how = "grid"
                break
        if how is None:
            ends = [float(theta_deriv(float(x))) for x in (lo, 0.5 * (lo + hi), hi)]
            n_ends_evals += 3
            sg = {_sign(v) for v in ends if math.isfinite(v)} - {0}
            if len(sg) > 1:
                how = "ends"
        if how is not None:
            rel = reg.relerr
            listed.append(
                {
                    "region_index": ri,
                    "how": how,
                    "start": float(reg.start),
                    "end": float(reg.end),
                    "depth": int(reg.depth),
                    "branch": "Levin" if reg.type == "Levin" else "CC",
                    "abserr": None if reg.abserr is None else float(reg.abserr),
                    "relerr": None if rel is None else float(rel),
                    "value_over_abserr": (
                        None if rel is None or rel == 0 else float(1.0 / rel)
                    ),
                    "phase_err": None if reg.phase_err is None else float(reg.phase_err),
                    "phase_limited": bool(reg.phase_limited),
                }
            )
    return {
        "n_grid": n,
        "h": h,
        "n_nonfinite": int((~finite).sum()),
        "n_sign_changes": len(changes),
        "min_over_max_abs_theta_deriv": ratio,
        "sign_change_x": change_x,
        "sign_change_halfwidth": change_halfwidth,
        "n_regions": len(regions),
        "n_regions_listed": len(listed),
        "n_regions_listed_grid": sum(1 for x in listed if x["how"] == "grid"),
        "n_regions_listed_ends": sum(1 for x in listed if x["how"] == "ends"),
        "listed": listed,
        "n_ends_evaluations": n_ends_evals,
        "seconds": time.perf_counter() - t0,
    }
