import os
"""
TK_07: impact of the BackgroundModel._build_derivative spline path on the two quantities the
transfer-function WKB branch actually consumes, Tk_omegaEff_sq and Tk_d_ln_omegaEff_dz.

Two ModelFunctions instances are built over the same Planck2018 LambdaCDM background and the
same z-grid as main.py (z in [0.1, 1e12], 100 samples per decade):
  EXACT  -- eps, eps', eps'', w', w'' from LambdaCDM's analytic methods (the code path taken
            when the cosmology supplies them: BackgroundModel.py:126-127)
  SPLINE -- the same quantities from the make_interp_spline/derivative stack
            (BackgroundModel.py:129-149), which is what LambdaCDM_GenericEOS gets
Then WKB_Tk.Tk_omegaEff_sq / Tk_d_ln_omegaEff_dz are evaluated with each.
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

from math import log

import numpy as np
from scipy.interpolate import make_interp_spline

from ComputeTargets.BackgroundModel import ModelFunctions
from ComputeTargets.WKB_Tk import Tk_d_ln_omegaEff_dz, Tk_omegaEff_sq
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

cosmo = LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())

Z_MIN, Z_MAX = 0.1, 1.0e12
n = int(round(100 * (np.log10(Z_MAX) - np.log10(Z_MIN))))
z_samples = np.expm1(np.linspace(log(1.0 + Z_MIN), log(1.0 + Z_MAX), n))
x_samples = np.log(1.0 + z_samples)


def build_derivative(y):
    raw = make_interp_spline(x_samples, y)
    d = raw.derivative()
    return np.array([float(d(xi)) / (1.0 + zi) for xi, zi in zip(x_samples, z_samples)])


def as_func(y):
    s = make_interp_spline(x_samples, y)
    return lambda z: float(s(log(1.0 + z)))


lnH = np.array([np.log(cosmo.Hubble(z)) for z in z_samples])
d1 = build_derivative(lnH)
d2 = build_derivative(d1)
d3 = build_derivative(d2)
wP = np.array([cosmo.wPerturbations(z) for z in z_samples])
wp1 = build_derivative(wP)
wp2 = build_derivative(wp1)


def make_model(exact: bool):
    if exact:
        f1, f2, f3 = cosmo.d_lnH_dz, cosmo.d2_lnH_dz2, cosmo.d3_lnH_dz3
        g1, g2 = cosmo.d_wPerturbations_dz, cosmo.d2_wPerturbations_dz2
    else:
        f1, f2, f3 = as_func(d1), as_func(d2), as_func(d3)
        g1, g2 = as_func(wp1), as_func(wp2)

    class M:
        pass

    m = M()
    m.functions = ModelFunctions(
        Hubble=cosmo.Hubble,
        epsilon=lambda z: (1.0 + z) * f1(z),
        d_epsilon_dz=lambda z: f1(z) + (1.0 + z) * f2(z),
        d2_epsilon_dz2=lambda z: 2.0 * f2(z) + (1.0 + z) * f3(z),
        wBackground=cosmo.wBackground,
        wPerturbations=cosmo.wPerturbations,
        tau=None,
        T_photon=cosmo.T_photon,
        d_lnH_dz=f1,
        d2_lnH_dz2=f2,
        d3_lnH_dz3=f3,
        d_wPerturbations_dz=g1,
        d2_wPerturbations_dz2=g2,
    )
    return m


exact, spline = make_model(True), make_model(False)

for k in [1.0, 1.0e3]:
    print(f"\n=== k/a0 = {k} (code units, 1/Mpc-ish) ===")
    print(f"  {'z':>12} {'omega_eff^2 rel diff':>21} {'d_ln_omega_dz rel diff':>23}")
    probes = [z_samples[i] for i in [0, 1, 2, 5, 20, n // 4, n // 2, 3 * n // 4, n - 21, n - 6, n - 3, n - 2, n - 1]]
    for z in probes:
        a = Tk_omegaEff_sq(exact, k, z)
        b = Tk_omegaEff_sq(spline, k, z)
        c = Tk_d_ln_omegaEff_dz(exact, k, z)
        d = Tk_d_ln_omegaEff_dz(spline, k, z)
        print(f"  {z:12.5g} {abs(b-a)/max(abs(a),1e-300):21.3e} {abs(d-c)/max(abs(c),1e-300):23.3e}")
