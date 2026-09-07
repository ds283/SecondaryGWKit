import os
"""GK_04: check the WKB matching implemented in ComputeTargets/GkWKBIntegration.py:389-418
against spec 02 R34/R36/R37 (= R25/R26/R30 of NUM 05).

(a) synthetic test: given an arbitrary quadruple (G*, G'*, omega*, omega'*, eps*), verify that
    the code's raw_cos_coeff / raw_sin_coeff reproduce G* and G'* when substituted into the
    Liouville-Green mode functions
        G(z) = omega^{-1/2}(z) (H*/H(z))^{1/2} [ cos_coeff cos Theta + sin_coeff sin Theta ]
    with Theta(z*) = 0, dTheta/dz = +omega_eff.
(b) verify the B sin(Theta + deltaTheta) rewriting (spec R37) is exact.
(c) end-to-end: on an exact constant-w background, integrate the code's numeric RHS from a
    subhorizon source, cut at z*, build the coefficients exactly as GkWKBIntegration.store()
    does, integrate the phase with the code's dTheta/dz = +omega_eff, and compare the resulting
    WKB Green's function with the exact solution.
"""

import sys
from math import sqrt, sin, cos, atan2, fabs
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.getcwd())  # run from the repository root

from ComputeTargets.GkNumericIntegration import RHS
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq, Gk_d_ln_omegaEff_dz
from ComputeTargets.analytic_Gk import compute_analytic_G

from GK_03_numeric_analytic import StubSupervisor, make_model


def code_coeffs(G, Gprime, omega_sq, d_ln_omega, eps, one_plus_z):
    """verbatim transcription of GkWKBIntegration.py:389-410"""
    omega = sqrt(omega_sq)
    sqrt_omega = sqrt(omega)
    raw_cos_coeff = sqrt_omega * G
    raw_sin_coeff = (Gprime + (G / 2.0) * (d_ln_omega + eps / one_plus_z)) / sqrt_omega
    deltaTheta = atan2(raw_cos_coeff, raw_sin_coeff)
    B = sqrt(raw_cos_coeff**2 + raw_sin_coeff**2)
    sgn_sin = +1 if sin(deltaTheta) >= 0.0 else -1
    sgn_G = +1 if G >= 0.0 else -1
    cos_coeff = 0.0
    sin_coeff = sgn_sin * sgn_G * B
    return raw_cos_coeff, raw_sin_coeff, deltaTheta, B, sin_coeff, cos_coeff


def synthetic():
    print("=== (a),(b) synthetic matching test")
    rng = np.random.default_rng(20260907)
    worst_G = worst_Gp = worst_shift = 0.0
    for trial in range(2000):
        Gs = rng.normal() * 10 ** rng.uniform(-3, 3)
        Gps = rng.normal() * 10 ** rng.uniform(-3, 3)
        omega = 10 ** rng.uniform(-1, 4)
        dlnomega = rng.normal() * omega * 0.1
        eps = rng.uniform(1.0, 2.5)
        z = rng.uniform(0.1, 100.0)
        opz = 1.0 + z

        rc, rs, dTheta, B, sin_c, cos_c = code_coeffs(
            Gs, Gps, omega**2, dlnomega, eps, opz
        )

        # mode functions of spec R34/R35 evaluated at z* (Theta=0, H*/H=1):
        #   G  = omega^{-1/2} (alpha cos 0 + beta sin 0) = alpha/omega^{1/2}
        G_rec = rc / sqrt(omega)
        # derivative at z*:  d/dz[ omega^{-1/2} f (a cos Theta + b sin Theta) ]
        #   = -(1/2)(omega'/omega) a/omega^{1/2} - (1/2)(eps/(1+z)) a/omega^{1/2} + b omega^{1/2}
        Gp_rec = (
            -0.5 * (dlnomega + eps / opz) * rc / sqrt(omega) + rs * sqrt(omega)
        )
        worst_G = max(worst_G, fabs(G_rec - Gs) / max(fabs(Gs), 1e-300))
        worst_Gp = max(worst_Gp, fabs(Gp_rec - Gps) / max(fabs(Gps), 1e-300))

        # (b) B sin(Theta + dTheta) == rc cos Theta + rs sin Theta for arbitrary Theta,
        #     and the sign-corrected sin_coeff must equal +B
        for Theta in rng.uniform(-10, 10, 5):
            lhs = sin_c * sin(Theta + dTheta) + cos_c * cos(Theta + dTheta)
            rhs = rc * cos(Theta) + rs * sin(Theta)
            worst_shift = max(worst_shift, fabs(lhs - rhs) / max(fabs(rhs), 1e-12))
    print(f"    worst rel. error reproducing G*      : {worst_G:.3e}")
    print(f"    worst rel. error reproducing G'*     : {worst_Gp:.3e}")
    print(f"    worst rel. error of B sin(T+dT) form : {worst_shift:.3e}")


def end_to_end(w=1.0 / 3.0, k=100.0, z_source=None, z_star=None):
    print(f"\n=== (c) end-to-end WKB reconstruction, w={w}, k={k}")
    model, b, n = make_model(w)
    sup = StubSupervisor()
    tau = model.functions.tau

    if z_source is None:
        z_source = 200.0
    if z_star is None:
        z_star = 150.0
    print(f"    k*tau(z_source={z_source})={k*tau(z_source):.4g}, k*tau(z*={z_star})={k*tau(z_star):.4g}")

    z_end = 5.0
    sol = solve_ivp(
        RHS,
        method="DOP853",
        t_span=(z_source, z_end),
        y0=[0.0, 1.0],
        args=(model, k, sup),
        rtol=1e-12,
        atol=1e-14,
        dense_output=True,
    )
    assert sol.success

    G_star, Gp_star = sol.sol(z_star)
    omega_sq_star = Gk_omegaEff_sq(model, k, z_star)
    dln_star = Gk_d_ln_omegaEff_dz(model, k, z_star)
    eps_star = model.functions.epsilon(z_star)
    H_star = model.functions.Hubble(z_star)

    rc, rs, dTheta, B, sin_c, cos_c = code_coeffs(
        G_star, Gp_star, omega_sq_star, dln_star, eps_star, 1.0 + z_star
    )
    print(
        f"    G*={G_star:.6g}, G'*={Gp_star:.6g}, omega*={sqrt(omega_sq_star):.6g}, "
        f"sin_coeff={sin_c:.6g}, cos_coeff={cos_c:.6g}, deltaTheta={dTheta:.6g}"
    )
    print(f"    sin_coeff == +B ? {sin_c > 0} (B={B:.6g})")

    # integrate the phase exactly as WKB_phase_function stage 1 does: dTheta/dz = +omega_eff
    zs = np.linspace(z_star, z_end, 400)
    phase = solve_ivp(
        lambda zz, st: [sqrt(Gk_omegaEff_sq(model, k, zz))],
        t_span=(z_star, z_end),
        y0=[0.0],
        t_eval=zs,
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    )
    assert phase.success

    print(f"    {'z':>8} {'theta':>12} {'G_WKB':>14} {'G_exact':>14} {'G_analytic':>14} {'rel':>10}")
    worst = 0.0
    for i in range(0, len(zs), 40):
        z = zs[i]
        theta = phase.y[0][i]
        omega = sqrt(Gk_omegaEff_sq(model, k, z))
        H = model.functions.Hubble(z)
        norm = sqrt((H_star / H) / omega)
        G_WKB = norm * (sin_c * sin(theta + dTheta) + cos_c * cos(theta + dTheta))
        G_exact = sol.sol(z)[0]
        G_an = compute_analytic_G(k, w, tau(z_source), tau(z), model.functions.Hubble(z_source))
        rel = fabs(G_WKB - G_exact) / max(fabs(G_exact), 1e-300)
        worst = max(worst, rel)
        print(f"    {z:8.3f} {theta:12.5g} {G_WKB:14.7g} {G_exact:14.7g} {G_an:14.7g} {rel:10.2e}")
    print(f"    worst |G_WKB/G_exact - 1| = {worst:.3e}")


if __name__ == "__main__":
    synthetic()
    end_to_end()
    end_to_end(w=0.2, k=100.0)
    # deep sub-horizon crossover (the pipeline hands over 3-6 e-folds inside the horizon)
    end_to_end(w=0.2, k=1.0e4)
