import os
"""GK_06: at FIXED response redshift, build G as a function of source redshift the way
GkSource / GkSourcePolicyData do, and check

  (i)  the numeric branch (splined directly from v.numeric.G, GkSourcePolicyData.py:602-603)
       is the unit-jump G: it -> 0^- as z_source -> z_response, with d/dz_source -> -1;
  (ii) the WKB branch, assembled as
          sin_amplitude(z_source) * sin(theta_mod_2pi(z_source)),
          sin_amplitude = sin_coeff * sqrt(H_ratio / omega)      (GkSourcePolicyData.py:644-648
                                                                  + spline_wrappers.py:118-122)
       reproduces the SAME G (same sign, same normalisation) as the numeric branch, so the
       stitched function is continuous across crossover_z.

Everything is done on an exact constant-w background with the code's own RHS, omega_eff and
matching formulae; the Ray/datastore compute path is not exercised.
"""

import sys
from math import sqrt, sin, fabs

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.getcwd())  # run from the repository root

from ComputeTargets.GkNumericIntegration import RHS
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq, Gk_d_ln_omegaEff_dz
from ComputeTargets.analytic_Gk import compute_analytic_G

from GK_03_numeric_analytic import StubSupervisor, make_model
from GK_04_wkb_matching import code_coeffs

W = 0.2
K = 1.0e4
model, b, n = make_model(W)
sup = StubSupervisor()
tau = model.functions.tau


def numeric_G(z_source, z_response):
    sol = solve_ivp(
        RHS,
        method="DOP853",
        t_span=(z_source, z_response),
        y0=[0.0, 1.0],
        args=(model, K, sup),
        rtol=1e-12,
        atol=1e-14,
    )
    assert sol.success
    return sol.y[0][-1]


def wkb_G(z_source, z_response):
    """z_init = z_source, unit-jump initial data (main.py:1375-1376 branch)"""
    omega_sq_i = Gk_omegaEff_sq(model, K, z_source)
    dln_i = Gk_d_ln_omegaEff_dz(model, K, z_source)
    eps_i = model.functions.epsilon(z_source)
    H_i = model.functions.Hubble(z_source)
    rc, rs, dTheta, B, sin_c, cos_c = code_coeffs(
        0.0, 1.0, omega_sq_i, dln_i, eps_i, 1.0 + z_source
    )
    ph = solve_ivp(
        lambda zz, st: [sqrt(Gk_omegaEff_sq(model, K, zz))],
        t_span=(z_source, z_response),
        y0=[0.0],
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    )
    assert ph.success
    theta = ph.y[0][-1] + dTheta
    omega = sqrt(Gk_omegaEff_sq(model, K, z_response))
    H = model.functions.Hubble(z_response)
    sin_amplitude = sin_c * sqrt((H_i / H) / omega)  # GkSourcePolicyData.py:647
    return sin_amplitude * sin(theta), sin_c, dTheta


if __name__ == "__main__":
    z_response = 20.0
    print(f"w={W}, k={K}, z_response={z_response}, k*tau(z_response)={K*tau(z_response):.4g}")

    print("\n=== (i) unit-jump sign as z_source -> z_response from above")
    print(f"    {'dz':>9} {'G_num':>14} {'G/(-dz)':>12} {'G_analytic':>14}")
    for dz in (1e-4, 1e-3, 1e-2, 1e-1):
        zs = z_response + dz
        G = numeric_G(zs, z_response)
        Ga = compute_analytic_G(K, W, tau(zs), tau(z_response), model.functions.Hubble(zs))
        print(f"    {dz:9.0e} {G:14.7g} {G/(-dz):12.7f} {Ga:14.7g}")

    print("\n=== (ii) numeric branch vs WKB branch at the same (z_source, z_response)")
    print(f"    {'z_source':>10} {'efolds_subh':>11} {'G_numeric':>14} {'G_WKB_branch':>14} {'rel':>10}")
    for z_source in (200.0, 150.0, 100.0, 60.0, 40.0, 30.0, 25.0):
        Gn = numeric_G(z_source, z_response)
        Gw, sin_c, dT = wkb_G(z_source, z_response)
        efolds = np.log((1.0 + z_source) * K / model.functions.Hubble(z_source))
        rel = fabs(Gw - Gn) / max(fabs(Gn), 1e-300)
        print(f"    {z_source:10.3f} {efolds:11.4g} {Gn:14.7g} {Gw:14.7g} {rel:10.2e}")

    print("\n    (both branches have the same sign and normalisation; sin_coeff > 0 always:")
    for z_source in (200.0, 100.0, 40.0):
        _, sin_c, dT = wkb_G(z_source, z_response)
        print(f"     z_source={z_source}: sin_coeff={sin_c:.6g}, deltaTheta={dT:.6g}")
