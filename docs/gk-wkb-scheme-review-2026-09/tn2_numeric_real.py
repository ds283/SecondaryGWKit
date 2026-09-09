"""GkNumericIntegration path on the real LambdaCDM background: cost, diagnostic overhead, agreement with the radiation oracle."""
import sys, time, numpy as np, mpmath as mp
from math import exp, log
from types import SimpleNamespace as NS
sys.path.insert(0, __file__.rsplit('/',1)[0])
from realbg import cosmo, model, H_mp
from common import grid
from CosmologyConcepts import redshift, redshift_array
from scipy.optimize import brentq
import ComputeTargets.GkNumericIntegration; GN = sys.modules["ComputeTargets.GkNumericIntegration"]
import Quadrature.integrators.numeric_with_phase_cut as N
from ComputeTargets.analytic_Gk import compute_analytic_G
N.check_units = lambda *a, **k: None
run = N.numeric_with_phase_cut._function
proxy = NS(get=lambda: model)
def tau_mp(z):  # conformal time from the big bang, int_z^inf dz/H
    f = lambda u: mp.exp(u)/H_mp(mp.exp(u)-1)
    return mp.quad(f, [mp.log(1+mp.mpf(z)), mp.log(1+mp.mpf(z))+5, mp.log(1+mp.mpf(z))+15, mp.log(1+mp.mpf(z))+40])
for k in [1e5, 3e8]:
    xof = lambda z: k*(1+z)/cosmo.Hubble(z)
    z_exit = brentq(lambda z: log(xof(z)), 1.0, 1e18)
    z_e3 = brentq(lambda z: log(xof(z))-3, 1.0, z_exit); z_e6 = brentq(lambda z: log(xof(z))-6, 1.0, z_exit)
    z_s = brentq(lambda z: log(xof(z))+5, z_exit, 1e19)
    zs = grid((1+z_s)/10**(12/100)-1, 0.85*z_e6, 100/12)
    zsamp = redshift_array([redshift(i+1, float(z)) for i, z in enumerate(zs)])
    kmock = NS(k=NS(k=k, k_inv_Mpc=k, store_id=0), z_exit=z_exit)
    res = {}
    for label, patch in [("with diagnostic", None), ("without diagnostic", lambda m, kk, z: 1.0)]:
        saved = GN.Gk_omegaEff_sq
        if patch: GN.Gk_omegaEff_sq = patch
        t0 = time.perf_counter()
        out = run(proxy, kmock, redshift(0, z_s), zsamp, 0.0, 1.0, GN.RHS, atol=1e-10, rtol=1e-8, delta_logz=0.01, mode="stop",
                  stop_search_window_z_begin=z_e3, stop_search_window_z_end=z_e6, task_label="tn2", object_label="G")
        res[label] = (time.perf_counter()-t0, out)
        GN.Gk_omegaEff_sq = saved
    (t1, out), (t2, _) = res["with diagnostic"], res["without diagnostic"]
    print(f"k={k:.0e}: z_source={z_s:.3g} (5 e-folds outside), {len(zs)} response samples to 0.85 z_e6, stop at z={z_exit-out['stop_deltaz_subh']:.4g} (x={xof(z_exit-out['stop_deltaz_subh']):.1f}); nfev={out['data'].RHS_evaluations}; time {t1:.2f}s with per-RHS omega diagnostic, {t2:.2f}s without; has_unresolved_osc={out['has_unresolved_osc']} at x={xof(out['unresolved_z']) if out['unresolved_z'] else None}")
    Hs = cosmo.Hubble(z_s); ts = tau_mp(z_s)
    errs = []
    for z, G in list(zip(zs, out["value_sample"]))[::max(1, len(out["value_sample"])//8)]:
        t = tau_mp(z); ref = compute_analytic_G(k, 1/3, float(ts), float(t), Hs)
        env = (1+z_s)**2/k  # radiation envelope in these units (H~s^2 normalisation differs: use ref scale)
        errs.append((z, xof(z), G, ref, (G-ref)/max(abs(ref), 1e-300)))
    print("   z, x, G_numeric, G_radiation_oracle, relative difference:")
    for e in errs: print(f"   {e[0]:10.4g} {e[1]:8.3g} {e[2]:14.6e} {e[3]:14.6e} {e[4]:10.2e}")
