import os
"""
TK_06: quantify the grid-end bias of BackgroundModel._build_derivative for a cosmology that
supplies NO analytic derivative methods (which is the case for LambdaCDM_GenericEOS: it defines
none of d_lnH_dz, d2_lnH_dz2, d3_lnH_dz3, d_wPerturbations_dz, d2_wPerturbations_dz2).

We drive the *same* code path (make_interp_spline in log(1+z), .derivative(), /(1+z)) on the
Planck2018 LambdaCDM background, where exact analytic derivatives are available for reference.
Grid: log-spaced z in [0.1, 1e12] at 100 samples per decade, matching main.py's
source_z_grid (samples_per_log10z=100, z_end=0.1).
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

from math import log

import numpy as np
from scipy.interpolate import make_interp_spline

from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

cosmo = LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())

Z_MIN, Z_MAX = 0.1, 1.0e12
n = int(round(100 * (np.log10(Z_MAX) - np.log10(Z_MIN))))
z_samples = np.expm1(np.linspace(log(1.0 + Z_MIN), log(1.0 + Z_MAX), n))
print(f"grid: n={n} samples, z in [{z_samples[0]:.4g}, {z_samples[-1]:.4g}]")


def build_derivative(y_samples):
    """BackgroundModel._build_derivative, lines 129-149 (spline branch)"""
    x = np.log(1.0 + z_samples)
    raw = make_interp_spline(x, y_samples)
    d = raw.derivative()
    return np.array([float(d(xi)) / (1.0 + zi) for xi, zi in zip(x, z_samples)])


lnH = np.array([np.log(cosmo.Hubble(z)) for z in z_samples])
d1 = build_derivative(lnH)
d2 = build_derivative(d1)
d3 = build_derivative(d2)

wP = np.array([cosmo.wPerturbations(z) for z in z_samples])
wp1 = build_derivative(wP)
wp2 = build_derivative(wp1)

opz = 1.0 + z_samples
eps_spl = opz * d1
eps1_spl = d1 + opz * d2
eps2_spl = 2.0 * d2 + opz * d3

eps_ex = np.array([opz[i] * cosmo.d_lnH_dz(z) for i, z in enumerate(z_samples)])
eps1_ex = np.array(
    [cosmo.d_lnH_dz(z) + opz[i] * cosmo.d2_lnH_dz2(z) for i, z in enumerate(z_samples)]
)
eps2_ex = np.array(
    [2.0 * cosmo.d2_lnH_dz2(z) + opz[i] * cosmo.d3_lnH_dz3(z) for i, z in enumerate(z_samples)]
)
wp1_ex = np.array([cosmo.d_wPerturbations_dz(z) for z in z_samples])
wp2_ex = np.array([cosmo.d2_wPerturbations_dz2(z) for z in z_samples])

print(f"\n{'quantity':>12} {'stack depth':>11} | relative error:"
      f" {'z=0.1 (end)':>13} {'2nd pt':>10} {'3rd pt':>10}"
      f" {'median':>10} {'z=1e12 (end)':>13} {'2nd from top':>13}")
for name, depth, got, want in [
    ("epsilon", 1, eps_spl, eps_ex),
    ("d_eps_dz", 2, eps1_spl, eps1_ex),
    ("d2_eps_dz2", 3, eps2_spl, eps2_ex),
    ("w'", 1, wp1, wp1_ex),
    ("w''", 2, wp2, wp2_ex),
]:
    r = np.abs((got - want) / want)
    print(f"{name:>12} {depth:>11} | {'':>15}"
          f" {r[0]:13.3e} {r[1]:10.3e} {r[2]:10.3e} {np.median(r[5:-5]):10.3e}"
          f" {r[-1]:13.3e} {r[-2]:13.3e}")
