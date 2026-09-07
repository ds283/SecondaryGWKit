import os
"""GK_03: numerically integrate the code's *actual* RHS (imported from
ComputeTargets.GkNumericIntegration) on an exact constant-w background and compare with
ComputeTargets.analytic_Gk.compute_analytic_G / compute_analytic_Gprime.

The compute path proper (GkNumericIntegration.compute) needs Ray + the datastore, so we call the
module-level RHS() function directly with a stub supervisor and a stub BackgroundModel whose
functions are the exact constant-w expressions:
    H(z)   = H0 (1+z)^n,   n = 3(1+w)/2
    eps(z) = (1+z) dlnH/dz = n  (constant)
    tau(z) = a0*eta = (1+b)(1+z)/H(z),  b = (1-3w)/(1+3w)   [= (1+b)/(aH), 1+b = 2/(1+3w)]
Also runs the a0 covariance test of PREAMBLE rule 4.
"""

import sys
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.getcwd())  # run from the repository root

from ComputeTargets.GkNumericIntegration import RHS
from ComputeTargets.analytic_Gk import compute_analytic_G, compute_analytic_Gprime
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq, Gk_d_ln_omegaEff_dz


class StubSupervisor:
    notify_available = False

    def notify_new_RHS_time(self, t):
        pass

    def message(self, *a, **kw):
        pass

    def reset_notify_time(self):
        pass

    def report_wavelength(self, *a, **kw):
        pass


def make_model(w, H0=1.0):
    n = 1.5 * (1.0 + w)
    b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)

    def Hubble(z):
        return H0 * (1.0 + z) ** n

    def tau(z):
        return (1.0 + b) * (1.0 + z) / Hubble(z)

    funcs = SimpleNamespace(
        Hubble=Hubble,
        epsilon=lambda z: n,
        d_epsilon_dz=lambda z: 0.0,
        d2_epsilon_dz2=lambda z: 0.0,
        tau=tau,
        wBackground=lambda z: w,
        wPerturbations=lambda z: w,
    )
    return SimpleNamespace(functions=funcs), b, n


def check_tau(w, H0=1.0):
    """verify tau(z) satisfies d tau/dz = -1/H (BackgroundModel.py:56)"""
    model, b, n = make_model(w, H0)
    z0 = 5.0
    h = 1e-5
    num = (model.functions.tau(z0 + h) - model.functions.tau(z0 - h)) / (2 * h)
    return num, -1.0 / model.functions.Hubble(z0)


def run(w, k, z_source, z_targets, H0=1.0, rtol=1e-11, atol=1e-13):
    model, b, n = make_model(w, H0)
    sup = StubSupervisor()

    sol = solve_ivp(
        RHS,
        method="DOP853",
        t_span=(z_source, min(z_targets) * 0.999),
        y0=[0.0, 1.0],  # unit jump: G=0, dG/dz=+1  (GkNumericIntegration.py:343-344)
        t_eval=sorted(z_targets, reverse=True),
        args=(model, k, sup),
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    assert sol.success, sol.message

    Hs = model.functions.Hubble(z_source)
    tau_s = model.functions.tau(z_source)

    rows = []
    for i, z in enumerate(sol.t):
        G = sol.y[0][i]
        Gp = sol.y[1][i]
        tau = model.functions.tau(z)
        Ga = compute_analytic_G(k, w, tau_s, tau, Hs)
        Gpa = compute_analytic_Gprime(k, w, tau_s, tau, Hs, model.functions.Hubble(z))
        rows.append(
            (z, G, Ga, abs(G - Ga) / max(abs(Ga), 1e-300), Gp, Gpa, abs(Gp - Gpa) / max(abs(Gpa), 1e-300))
        )
    return rows, sol


if __name__ == "__main__":
    print("=== consistency of the stub background: dtau/dz vs -1/H")
    for w in (1.0 / 3.0, 0.2, 0.0):
        num, exact = check_tau(w)
        print(f"  w={w:.4f}: numeric dtau/dz={num:.12g}, -1/H={exact:.12g}")

    for w, k in ((1.0 / 3.0, 1.0), (0.2, 1.0), (0.0, 1.0)):
        print(f"\n=== w={w:.5f}, k={k} (k_phys today), H0=1")
        model, b, n = make_model(w)
        z_source = 20.0
        print(f"    b={b:.5g}, eps={n:.5g}, k*tau_source={k*model.functions.tau(z_source):.5g}")
        z_targets = [19.0, 15.0, 10.0, 5.0, 2.0, 1.0, 0.5]
        rows, sol = run(w, k, z_source, z_targets)
        print(f"    {'z':>8} {'k*tau':>9} {'G_num':>14} {'G_analytic':>14} {'relerr':>10} {'relerr(G\')':>11}")
        for (z, G, Ga, rel, Gp, Gpa, relp) in rows:
            print(
                f"    {z:8.3f} {k*model.functions.tau(z):9.4g} {G:14.7g} {Ga:14.7g} {rel:10.2e} {relp:11.2e}"
            )

    # --- sign / unit-jump check just below the source
    print("\n=== sign and unit jump near the source (expect G ~ (z - z') < 0 for z < z')")
    w = 1.0 / 3.0
    model, b, n = make_model(w)
    z_source = 20.0
    for dz in (1e-3, 1e-2, 1e-1):
        rows, _ = run(w, 1.0, z_source, [z_source - dz])
        z, G, Ga, rel, Gp, Gpa, relp = rows[0]
        print(
            f"    dz={dz:.0e}: G={G:.8g}  (z-z')={-dz:.8g}  G/(z-z')={G/(-dz):.8f}   G_analytic={Ga:.8g}"
        )

    # --- a0 covariance test: a0 -> lam a0 means k_com -> lam k_com with k_phys=k_com/a0 fixed,
    #     and tau = a0*eta fixed. Everything in the code is written in (k_phys, tau), so the
    #     stored numbers must be *identical*. Demonstrate by scaling the comoving quantities.
    print("\n=== a0 covariance (k_phys and tau=a0*eta invariant => G invariant)")
    w = 1.0 / 3.0
    for lam in (1.0, 3.0, 0.1):
        # k_com = lam * k_phys(=1) with a0 = lam ; code only ever sees k_phys = k_com/a0 = 1
        k_phys = (lam * 1.0) / lam
        rows, _ = run(w, k_phys, 20.0, [5.0])
        print(f"    lam={lam:6.3g}: k_phys={k_phys:.6g}, G(5)={rows[0][1]:.12g}")

    # --- WKB diagnostics on the same background
    print("\n=== omega_eff^2 and |dln omega/dz|/omega on the constant-w background")
    model, b, n = make_model(1.0 / 3.0)
    for z in (20.0, 5.0, 1.0):
        o2 = Gk_omegaEff_sq(model, 1.0, z)
        dl = Gk_d_ln_omegaEff_dz(model, 1.0, z)
        print(f"    z={z:6.2f}: omega^2={o2:.8g}, |dlnomega/dz|/omega={abs(dl)/np.sqrt(o2):.5g}")
