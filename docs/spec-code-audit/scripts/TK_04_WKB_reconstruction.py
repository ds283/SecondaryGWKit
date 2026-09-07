import os
"""
TK_04: reproduce the Liouville-Green reconstruction of T_k exactly as
ComputeTargets/TkWKBIntegration.store() (lines 434-506) assembles it, on a constant-w
background, and compare with the exact Bessel solution (spec 01 R11 =
ComputeTargets/analytic_Tk.compute_analytic_T).

The code's ingredients are used unmodified:
  * omega_eff^2         : ComputeTargets/WKB_Tk.Tk_omegaEff_sq
  * d ln omega_eff / dz : ComputeTargets/WKB_Tk.Tk_d_ln_omegaEff_dz
  * friction RHS        : ComputeTargets/TkWKBIntegration.friction_RHS
  * phase               : d theta / dz = +sqrt(omega_eff^2), theta(z_init)=0
                          (Quadrature/integrators/WKB_phase_function.py:96-97,101)
  * coefficients        : TkWKBIntegration.store() raw_cos_coeff / raw_sin_coeff / deltaTheta
  * amplitude           : sqrt((H_init/H)/omega) * exp(friction)
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

from math import atan2, cos, exp, sin, sqrt

import numpy as np
from scipy.integrate import solve_ivp

from ComputeTargets.TkWKBIntegration import friction_RHS
from ComputeTargets.WKB_Tk import Tk_d_ln_omegaEff_dz, Tk_omegaEff_sq
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime

from TK_03_numeric_vs_analytic import DummySupervisor, FakeModel


def wkb_reconstruct(w, k, z_init, z_stop, n=25):
    model = FakeModel(w)
    sup = DummySupervisor()

    # exact initial data taken from the analytic solution at z_init (this is what the pipeline
    # hands over from the numeric integration at the phase-cut point)
    H_init = model.Hubble(z_init)
    T_init = compute_analytic_T(k, w, model.tau(z_init))
    Tprime_init = compute_analytic_Tprime(k, w, model.tau(z_init), H_init)

    z_eval = np.exp(np.linspace(np.log(1.0 + z_init), np.log(1.0 + z_stop), n)) - 1.0
    z_eval[0] = z_init
    z_eval[-1] = z_stop

    # phase: WKB_phase_function stage 1, dtheta/dz = +omega, theta(z_init) = 0
    sol_theta = solve_ivp(
        lambda z, y: [sqrt(Tk_omegaEff_sq(model, k, z))],
        t_span=(z_init, z_stop),
        y0=[0.0],
        t_eval=z_eval,
        method="DOP853",
        atol=1e-13,
        rtol=1e-13,
    )
    assert sol_theta.success

    # friction: integrate_friction_function, initial state 0.0 at z_init
    sol_fric = solve_ivp(
        lambda z, y: friction_RHS(z, list(y), model, k, sup),
        t_span=(z_init, z_stop),
        y0=[0.0],
        t_eval=z_eval,
        method="DOP853",
        atol=1e-13,
        rtol=1e-13,
    )
    assert sol_fric.success

    # --- TkWKBIntegration.store() STEP 1/2 -----------------------------------
    eps_init = model.functions.epsilon(z_init)
    cs2_init = model.functions.wPerturbations(z_init)
    omega_sq_init = Tk_omegaEff_sq(model, k, z_init)
    d_ln_omega_init = Tk_d_ln_omegaEff_dz(model, k, z_init)
    omega_init = sqrt(omega_sq_init)
    sqrt_omega_init = sqrt(omega_init)
    one_plus_z_init = 1.0 + z_init

    raw_cos_coeff = sqrt_omega_init * T_init
    raw_sin_coeff = (
        Tprime_init
        + (T_init / 2.0)
        * (d_ln_omega_init + (eps_init - 3.0 * (1.0 + cs2_init)) / one_plus_z_init)
    ) / sqrt_omega_init

    deltaTheta = atan2(raw_cos_coeff, raw_sin_coeff)
    B = sqrt(raw_cos_coeff**2 + raw_sin_coeff**2)
    sgn = (+1 if sin(deltaTheta) >= 0.0 else -1) * (+1 if T_init >= 0.0 else -1)
    sin_coeff = sgn * B

    print(f"\n=== w={w}, k={k}, z_init={z_init} (k/aH_init = "
          f"{(1+z_init)*k/H_init:.4g}) ===")
    print(f"  sgn correction factor = {sgn:+d}  (expected +1: it is always a no-op)")
    print(f"  {'z':>12} {'theta':>12} {'T_WKB':>15} {'T_exact':>15} {'rel diff':>10}"
          f" {'err/envelope':>13}")
    worst = 0.0
    worst_env = 0.0
    for i, z in enumerate(sol_theta.t):
        theta = sol_theta.y[0][i] + deltaTheta
        H = model.Hubble(z)
        omega = sqrt(Tk_omegaEff_sq(model, k, z))
        norm = sqrt((H_init / H) / omega)
        envelope = norm * exp(sol_fric.y[0][i]) * abs(sin_coeff)
        T_wkb = norm * exp(sol_fric.y[0][i]) * sin_coeff * sin(theta)
        T_ex = compute_analytic_T(k, w, model.tau(z))
        rel = abs(T_wkb - T_ex) / max(abs(T_ex), 1e-300)
        env = abs(T_wkb - T_ex) / envelope
        if i > 0:
            worst = max(worst, rel)
            worst_env = max(worst_env, env)
        if i % max(1, len(sol_theta.t) // 8) == 0 or i == len(sol_theta.t) - 1:
            print(f"  {z:12.5g} {theta:12.5g} {T_wkb:15.7g} {T_ex:15.7g} {rel:10.2e}"
                  f" {env:13.2e}")
    print(f"  worst (excluding z_init): rel diff {worst:.3e},  |err|/envelope {worst_env:.3e}")
    return worst, worst_env


def z_at_efolds_subh(w, k, n_efolds):
    """z where k/(aH) = exp(n_efolds); k/(aH) = k (1+z)^(1-p)"""
    p = 1.5 * (1.0 + w)
    return (k / np.exp(n_efolds)) ** (1.0 / (p - 1.0)) - 1.0


if __name__ == "__main__":
    # start 3 e-folds inside the horizon, as the pipeline's numeric->WKB handover does
    # (TkNumericIntegration: search window z_exit_subh_e3 .. z_exit_subh_e6)
    for w in [1.0 / 3.0, 0.2, 0.5]:
        for k in [1.0e4, 1.0e6]:
            wkb_reconstruct(
                w,
                k=k,
                z_init=z_at_efolds_subh(w, k, 3.0),
                z_stop=z_at_efolds_subh(w, k, 8.0),
            )
