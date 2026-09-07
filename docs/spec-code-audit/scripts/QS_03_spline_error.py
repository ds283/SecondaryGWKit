import os
"""QS_03: quantify the F1 issue (docs/resonance-scaffolding/sigw-resonance-reconciliation.md
section 0.2) -- QuadSource._create_functions (ComputeTargets/QuadSource.py:294-309)
fits a plain cubic make_interp_spline through the *sampled* source values in
log(1+z).  Measure how well that spline reproduces the exact analytic radiation
source between nodes, on the code's own grid
(CosmologyConcepts/wavenumber.populate_z_sample: logspace in z,
DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z = 100 samples per decade of z,
main.py:69, main.py:321).

Exact radiation background, a0 absorbed: H = H0(1+z)^2, tau = a0*eta = 1/(H0(1+z)).
f is built with the SAME source_function the code uses, from the analytic_Tk oracle.
q = r = k (the deep sub-horizon diagonal pair).
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

import numpy as np
from scipy.interpolate import make_interp_spline

from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from ComputeTargets.QuadSource import source_function

H0 = 1.0
W = 1.0 / 3.0
CS = np.sqrt(W)
SAMPLES_PER_LOG10Z = 100  # main.py:69
Z_INIT = 1.0e8
Z_END = 0.1  # main.py:71 DEFAULT_ZEND


def tau(z):
    return 1.0 / (H0 * (1.0 + z))


def H(z):
    return H0 * (1.0 + z) ** 2


def f_exact(z, q, r):
    t, h = tau(z), H(z)
    return source_function(
        compute_analytic_T(q, W, t),
        compute_analytic_T(r, W, t),
        compute_analytic_Tprime(q, W, t, h),
        compute_analytic_Tprime(r, W, t, h),
        z,
        W,
    )["source"]


def code_grid():
    n = int(round(SAMPLES_PER_LOG10Z * (np.log10(Z_INIT) - np.log10(Z_END)) + 0.5, 0))
    return np.logspace(np.log10(Z_INIT), np.log10(Z_END), num=n)


zs = code_grid()
print(f"grid: {len(zs)} points, z in [{zs[-1]:.4g}, {zs[0]:.4g}]")
print(f"max node spacing in log(1+z) = {np.max(np.diff(np.log(1.0+np.sort(zs)))):.4f}\n")

print("k chosen so that x_end = k*c_s*tau(z_end) takes the listed value.")
print("'cycles' = total oscillation cycles of f (= x_end/pi, f oscillates at 2*theta_q).")
print("err_env = max |spline - exact| between nodes, normalised to the local")
print("oscillation envelope of f at that z (a fair measure for an oscillation).\n")
print(f"{'x_end':>9} {'k':>11} {'cycles':>10} {'pts/half-cyc @x_end':>20} "
      f"{'max err_env':>12} {'at z':>10} {'max |dspline/f|':>16}")

for x_end in (3.0, 10.0, 30.0, 100.0, 300.0, 1.0e3, 1.0e4):
    k = x_end / (CS * tau(Z_END))
    y = np.array([f_exact(z, k, k) for z in zs])
    order = np.argsort(np.log(1.0 + zs))
    xs = np.log(1.0 + zs[order])
    spl = make_interp_spline(xs, y[order])  # exactly as QuadSource.py:299

    # dense sample: 41 points inside every node interval
    worst_env, worst_z, worst_rel = 0.0, None, 0.0
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        xd = xa + (xb - xa) * np.linspace(0.0, 1.0, 41)[1:-1]
        zd = np.exp(xd) - 1.0
        fe = np.array([f_exact(z, k, k) for z in zd])
        fs = spl(xd)
        # local envelope: max |f| over this node interval plus its neighbours
        lo, hi = max(0, i - 2), min(len(xs), i + 3)
        env = max(np.max(np.abs(y[order][lo:hi])), np.max(np.abs(fe)))
        e = np.max(np.abs(fs - fe)) / max(env, 1e-300)
        if e > worst_env:
            worst_env, worst_z = e, zd[int(np.argmax(np.abs(fs - fe)))]
        r = np.max(np.abs(fs - fe) / np.maximum(np.abs(fe), 1e-300))
        worst_rel = max(worst_rel, r)

    dx = np.max(np.diff(xs))
    pts_per_half = np.pi / (dx * x_end) if x_end > 0 else np.inf
    print(f"{x_end:9.3g} {k:11.4g} {x_end/np.pi:10.4g} {pts_per_half:20.3g} "
          f"{worst_env:12.3e} {worst_z:10.3g} {worst_rel:16.3e}")

print("\n(pts/half-cyc: number of grid nodes per half oscillation of f at z_end;")
print(" a cubic interpolant needs >~ 10 to be accurate, and Nyquist fails below 2.)")
