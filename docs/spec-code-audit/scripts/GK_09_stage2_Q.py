import os
"""GK_09: verify the stage-2 phase representation of spec 02 R38 as implemented in
Quadrature/integrators/WKB_phase_function.py:262-303:

    u = z_init - z,  Theta(z) = Theta_i + omega_i (1+u) Q(u),
    dQ/du = -(omega/omega_i)/(1+u) - Q/(1+u),   Q(0) = 0

against a direct integration of dTheta/dz = +omega_eff.  Also records the sign of Q (spec 02
sign-off 2.8 / Q10: the fixed point is Q = -1, "close to unity" in magnitude).
"""

import sys
from math import sqrt, fabs

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.getcwd())  # run from the repository root

from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from GK_03_numeric_analytic import make_model

for w, k in ((1.0 / 3.0, 1.0e4), (0.2, 1.0e4)):
    model, b, n = make_model(w)
    omega = lambda zz: sqrt(Gk_omegaEff_sq(model, k, zz))

    z_init, z_end = 300.0, 5.0
    omega_i = omega(z_init)

    us = np.linspace(0.0, z_init - z_end, 500)
    Qsol = solve_ivp(
        lambda u, st: [-omega(z_init - u) / omega_i / (1.0 + u) - st[0] / (1.0 + u)],
        t_span=(0.0, z_init - z_end),
        y0=[0.0],
        t_eval=us,
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    )
    Tsol = solve_ivp(
        lambda zz, st: [omega(zz)],
        t_span=(z_init, z_end),
        y0=[0.0],
        t_eval=[z_init - u for u in us],
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    )
    theta_Q = omega_i * (1.0 + us) * Qsol.y[0]
    theta_direct = Tsol.y[0]
    rel = np.abs(theta_Q - theta_direct) / np.maximum(np.abs(theta_direct), 1e-12)
    print(f"w={w:.5f}, k={k}: omega_i={omega_i:.6g}")
    print(
        f"    max |theta_Q/theta_direct - 1| (u>1) = {np.max(rel[us > 1.0]):.3e}; "
        f"theta(z_end)={theta_direct[-1]:.6g}"
    )
    print(
        f"    Q range = [{Qsol.y[0].min():.6g}, {Qsol.y[0].max():.6g}]; "
        f"Q at u>>1: {Qsol.y[0][-1]:.6g}  (spec: fixed point Q=-1)"
    )
