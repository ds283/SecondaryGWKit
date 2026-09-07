import os
"""
TK_05: exercise the real LambdaCDM(Planck2018) model.

(a) spec 01 R15/R16/R28/R31 vs LambdaCDM.d_lnH_dz / wBackground / wPerturbations /
    d_wPerturbations_dz / d2_wPerturbations_dz2  (independent sympy-differentiated reference)
(b) spec 01 R18/R19/R20 vs ComputeTargets/BackgroundModel initial condition
    tau_init = sqrt(3) M_P (1+z_init)/sqrt(rho_init)  and  d(a0 tau)/dz = -1/H
(c) the spline path in BackgroundModel._build_derivative: reproduce it verbatim on a log(1+z)
    grid and measure the error at the grid ends against the analytic derivatives that the
    model itself supplies.
(d) the wPerturbations denominator discrepancy between LambdaCDM and LambdaCDM_GenericEOS.
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

from math import log, sqrt

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline

from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

units = Mpc_units()
params = Planck2018()
cosmo = LambdaCDM(store_id=0, units=units, params=params)

Om, Or, Occ = cosmo.omega_m, cosmo.omega_r, cosmo.omega_cc
print(f"\nOmega_m={Om:.6g}, Omega_r={Or:.6g}, Omega_cc={Occ:.6g}, H0={cosmo.H0:.6g}")

# ---------------------------------------------------------------- (a) sympy reference
zs = sp.Symbol("z")
E2 = Om * (1 + zs) ** 3 + Or * (1 + zs) ** 4 + Occ  # H^2/H0^2
lnH = sp.log(sp.sqrt(E2))
wB = (sp.Rational(1, 3) * Or * (1 + zs) ** 4 - Occ) / E2
wP = (sp.Rational(1, 3) * Or * (1 + zs)) / (Om + Or * (1 + zs))
ref = {
    "d_lnH_dz": sp.lambdify(zs, sp.diff(lnH, zs)),
    "d2_lnH_dz2": sp.lambdify(zs, sp.diff(lnH, zs, 2)),
    "d3_lnH_dz3": sp.lambdify(zs, sp.diff(lnH, zs, 3)),
    "wBackground": sp.lambdify(zs, wB),
    "wPerturbations": sp.lambdify(zs, wP),
    "d_wPerturbations_dz": sp.lambdify(zs, sp.diff(wP, zs)),
    "d2_wPerturbations_dz2": sp.lambdify(zs, sp.diff(wP, zs, 2)),
}

print("\n(a) LambdaCDM analytic methods vs sympy reference (max rel diff over z grid)")
z_grid = np.concatenate([np.array([0.0, 0.5, 2.0]), np.logspace(1, 14, 30)])
for name, fn in ref.items():
    worst = 0.0
    at = None
    for z in z_grid:
        a = getattr(cosmo, name)(float(z))
        b = float(fn(float(z)))
        d = abs(a - b) / max(abs(b), 1e-300)
        if d > worst:
            worst, at = d, z
    print(f"  {name:26s} max rel diff = {worst:.3e}  (at z={at:.4g})")

# ---------------------------------------------------------------- (b) tau initial condition
print("\n(b) spec 01 R20 initial condition for a0*tau, vs the exact R19 integral")
print(f"  {'z_init':>10} {'code tau_init':>16} {'exact int':>16} {'rel diff':>10}")
for z_init in [1.0e6, 1.0e8, 1.0e10, 1.0e12]:
    rho_init = cosmo.rho(z_init)
    tau_code = sqrt(3.0) * units.PlanckMass / sqrt(rho_init) * (1.0 + z_init)
    # exact:  a0 tau(z) = int_z^inf dz'/H(z')
    val, err = quad(
        lambda u: 1.0 / cosmo.Hubble(np.expm1(u)) * np.exp(u),
        log(1.0 + z_init),
        log(1.0 + 1.0e30),
        limit=400,
    )
    print(f"  {z_init:10.3g} {tau_code:16.9g} {val:16.9g} {abs(tau_code-val)/val:10.2e}")
print("  (also: (1+z)/H(z) == sqrt(3) M_P (1+z)/sqrt(rho) identically, since rho = 3 H^2 M_P^2:")
z_chk = 1.0e10
print(f"   at z={z_chk:g}, difference ="
      f" {abs(sqrt(3.0)*units.PlanckMass/sqrt(cosmo.rho(z_chk))*(1.0+z_chk) - (1.0+z_chk)/cosmo.Hubble(z_chk)):.3e})")

# ---------------------------------------------------------------- (c) spline derivative path
print("\n(c) BackgroundModel._build_derivative spline path: error vs analytic derivative")


def spline_deriv(f, z_samples):
    """verbatim reproduction of BackgroundModel._build_derivative for the branch where the
    cosmology does NOT supply an analytic method (BackgroundModel.py:129-149)"""
    data = [(log(1.0 + z), f(z)) for z in z_samples]
    data.sort(key=lambda pair: pair[0])
    x_data, y_data = zip(*data)
    raw = make_interp_spline(x_data, y_data)
    deriv = raw.derivative()
    return [float(deriv(log(1.0 + z))) / (1.0 + z) for z in z_samples]


for n in [200, 500, 1000]:
    z_samples = list(np.expm1(np.linspace(log(1.0 + 0.1), log(1.0 + 1.0e12), n)))
    got = spline_deriv(lambda z: log(cosmo.Hubble(z)), z_samples)
    want = [cosmo.d_lnH_dz(z) for z in z_samples]
    rel = np.abs((np.array(got) - np.array(want)) / np.array(want))
    print(f"  n={n:5d} d_lnH_dz: rel err at max-z end = {rel[-1]:.3e},"
          f" at min-z end = {rel[0]:.3e}, second/penultimate = {rel[1]:.3e}/{rel[-2]:.3e},"
          f" interior median = {np.median(rel[3:-3]):.3e}")

# and the *stacked* derivative (d2 from the d1 samples), as the code does when the model
# has no analytic method
print("\n  stacked derivative (d2_lnH_dz2 from d_lnH_dz samples):")
for n in [500, 1000]:
    z_samples = list(np.expm1(np.linspace(log(1.0 + 0.1), log(1.0 + 1.0e12), n)))
    d1 = spline_deriv(lambda z: log(cosmo.Hubble(z)), z_samples)
    d1_interp = dict(zip(z_samples, d1))
    d2 = spline_deriv(lambda z: d1_interp[z], z_samples)
    want = np.array([cosmo.d2_lnH_dz2(z) for z in z_samples])
    rel = np.abs((np.array(d2) - want) / want)
    print(f"  n={n:5d} rel err at max-z end = {rel[-1]:.3e}, at min-z end = {rel[0]:.3e},"
          f" interior median = {np.median(rel[3:-3]):.3e}")

# ---------------------------------------------------------------- (d) wPerturbations denominators
print("\n(d) wPerturbations: LambdaCDM (Lambda excluded from denominator) vs")
print("    LambdaCDM_GenericEOS (denominator = self.rho(z), Lambda INCLUDED)")
print(f"  {'z':>8} {'w_P (LambdaCDM)':>18} {'w_P (GenericEOS form)':>22} {'ratio':>8}")
for z in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
    wp_ok = cosmo.wPerturbations(z)
    rho = cosmo.rho(z)
    rho_r = cosmo.rho_r0 * (1.0 + z) ** 4
    wp_eos = (1.0 / 3.0) * rho_r / rho
    print(f"  {z:8.4g} {wp_ok:18.8g} {wp_eos:22.8g} {wp_ok/wp_eos:8.4f}")
