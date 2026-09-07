import os
"""
TK_03: integrate the *actual* RHS coded in ComputeTargets/TkNumericIntegration.py on a
constant-w background stand-in, and compare with ComputeTargets/analytic_Tk.compute_analytic_T
(spec 01 R11) and compute_analytic_Tprime.

The compute path proper (numeric_with_phase_cut / BackgroundModel) needs Ray + a Datastore, so
we drive the imported RHS function directly with a stand-in `model` object exposing the same
ModelFunctions interface.  Nothing is reimplemented: RHS, Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz,
compute_analytic_T, compute_analytic_Tprime are the repository functions.
"""

import sys

sys.path.insert(0, os.getcwd())  # run from the repository root

import numpy as np
from scipy.integrate import solve_ivp

from ComputeTargets.BackgroundModel import ModelFunctions
from ComputeTargets.TkNumericIntegration import RHS
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime


class DummySupervisor:
    notify_available = False

    def message(self, *a, **kw):
        pass

    def reset_notify_time(self):
        pass

    def notify_new_RHS_time(self, t):
        pass

    def report_wavelength(self, *a, **kw):
        pass


class FakeModel:
    """constant-w background: H = H0 (1+z)^p, p = 3(1+w)/2, eps = p, a0*tau = 1/(H0 (p-1) (1+z)^(p-1))"""

    def __init__(self, w, H0=1.0):
        self.w = w
        self.H0 = H0
        self.p = 1.5 * (1.0 + w)
        self.functions = ModelFunctions(
            Hubble=self.Hubble,
            epsilon=lambda z: self.p,
            d_epsilon_dz=lambda z: 0.0,
            d2_epsilon_dz2=lambda z: 0.0,
            wBackground=lambda z: self.w,
            wPerturbations=lambda z: self.w,
            tau=self.tau,
            T_photon=lambda z: 0.0,
            d_lnH_dz=lambda z: self.p / (1.0 + z),
            d2_lnH_dz2=lambda z: -self.p / (1.0 + z) ** 2,
            d3_lnH_dz3=lambda z: 2.0 * self.p / (1.0 + z) ** 3,
            d_wPerturbations_dz=lambda z: 0.0,
            d2_wPerturbations_dz2=lambda z: 0.0,
        )

    def Hubble(self, z):
        return self.H0 * (1.0 + z) ** self.p

    def tau(self, z):
        return 1.0 / (self.H0 * (self.p - 1.0) * (1.0 + z) ** (self.p - 1.0))


def run(w, k, z_init, z_stop, n=40):
    model = FakeModel(w)
    sup = DummySupervisor()

    z_eval = np.exp(np.linspace(np.log(1.0 + z_init), np.log(1.0 + z_stop), n)) - 1.0
    z_eval[0] = z_init

    sol = solve_ivp(
        lambda z, y: RHS(z, list(y), model, k, sup),
        t_span=(z_init, z_stop),
        y0=[1.0, 0.0],
        t_eval=z_eval,
        method="DOP853",
        atol=1e-12,
        rtol=1e-12,
    )
    assert sol.success, sol.message

    print(f"\n=== w = {w}, k = {k}, z_init = {z_init} ===")
    print(f"  k/(aH) at z_init = {(1+z_init)*k/model.Hubble(z_init):.4g}"
          f"  (e-folds superhorizon = {-np.log((1+z_init)*k/model.Hubble(z_init)):.3g})")
    print(f"  {'z':>12} {'k c_s tau':>11} {'T_numeric':>14} {'T_analytic':>14} {'rel diff':>10}"
          f" {'Tp_num':>13} {'Tp_analytic':>13} {'rel diff':>10}")
    worst_T = 0.0
    worst_Tp = 0.0
    for i, z in enumerate(sol.t):
        tau = model.tau(z)
        Ta = compute_analytic_T(k, w, tau)
        Tpa = compute_analytic_Tprime(k, w, tau, model.Hubble(z))
        Tn = sol.y[0][i]
        Tpn = sol.y[1][i]
        rT = abs(Tn - Ta) / max(abs(Ta), 1e-300)
        rTp = abs(Tpn - Tpa) / max(abs(Tpa), 1e-300)
        if i > 0:  # skip z_init, where T'=0 exactly by construction
            worst_T = max(worst_T, rT)
            worst_Tp = max(worst_Tp, rTp)
        if i % max(1, len(sol.t) // 10) == 0 or i == len(sol.t) - 1:
            print(f"  {z:12.5g} {k*np.sqrt(w)*tau:11.4g} {Tn:14.7g} {Ta:14.7g} {rT:10.2e}"
                  f" {Tpn:13.6g} {Tpa:13.6g} {rTp:10.2e}")
    print(f"  worst relative difference: T {worst_T:.3e},  dT/dz {worst_Tp:.3e}")
    return worst_T, worst_Tp


if __name__ == "__main__":
    # radiation, k such that horizon entry happens well inside the integration range
    run(1.0 / 3.0, k=1.0, z_init=1.0e4, z_stop=1.0)
    run(1.0 / 3.0, k=100.0, z_init=1.0e6, z_stop=1.0e2)
    # generic constant w
    run(0.2, k=1.0, z_init=1.0e4, z_stop=1.0)
    run(0.5, k=1.0, z_init=1.0e4, z_stop=1.0)
    # deep sub-horizon (many oscillations): k c_s tau up to ~O(50)
    run(1.0 / 3.0, k=1.0e4, z_init=1.0e6, z_stop=1.0e2)
    run(0.2, k=1.0e4, z_init=1.0e6, z_stop=1.0e2)

    # convergence: the residual is initial-data error, O((k c_s tau_init)^2).
    # Push z_init back and check the plateau falls as (1+z_init)^-2 for radiation.
    print("\n=== initial-data convergence, w=1/3, k=1e4, z_stop=100 ===")
    print(f"  {'z_init':>10} {'k c_s tau_init':>15} {'worst rel diff in T':>21}")
    prev = None
    for zi in [1.0e6, 4.0e6, 1.6e7, 6.4e7]:
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wT, _ = run(1.0 / 3.0, k=1.0e4, z_init=zi, z_stop=1.0e2)
        m = FakeModel(1.0 / 3.0)
        x = 1.0e4 * np.sqrt(1.0 / 3.0) * m.tau(zi)
        ratio = "" if prev is None else f"   (ratio to previous = {wT/prev:.3f})"
        print(f"  {zi:10.4g} {x:15.4g} {wT:21.3e}{ratio}")
        prev = wT
