"""Production phase solver (integrate_phase_function) on exact radiation: error vs span, tolerance, grid."""
import sys, time, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq as om2, Gk_d_ln_omegaEff_dz as dlom
import Quadrature.integrators.WKB_phase_function as W
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq as om2, Gk_d_ln_omegaEff_dz as dlom
def run(k, x_init, per_decade, rtol, atol, force_stage1=False, z_lo=0.1):
    zi = k/x_init - 1.0
    z_first = (1+zi)/10**(1/per_decade) - 1
    zs = grid(z_first, z_lo, per_decade)
    model = RadModel(); meta = {}
    saved = W.DEFAULT_OMEGA_WKB_SQ_MAX
    if force_stage1: W.DEFAULT_OMEGA_WKB_SQ_MAX = 1e300
    t0 = time.perf_counter()
    try:
        out = W.integrate_phase_function(model, key(k), zi, zarr(zs), om2, dlom, om2(model, k, zi), atol, rtol, meta, "t2", "G")
    finally:
        W.DEFAULT_OMEGA_WKB_SQ_MAX = saved
    dt = time.perf_counter() - t0
    err = phase_err(out["theta_div_2pi_sample"], out["theta_mod_2pi_sample"], k, zi, zs)
    span = float(abs(exact_theta(k, zi, z_lo)))
    n1 = out["stage_1_data"].RHS_evaluations if out["stage_1_data"] else 0
    n2 = out["stage_2_data"].RHS_evaluations if out["stage_2_data"] else 0
    return dict(span=span, maxerr=float(np.max(np.abs(err))), enderr=float(err[-1]), n1=n1, n2=n2,
                resets=meta.get("phase_cycle_events", 0), Qmin=meta.get("stage_2_smallest_Q"), nsamp=len(zs), dt=dt)
print(f"{'span':>8} {'grid/dec':>8} {'rtol':>7} {'atol':>7} {'stage1?':>7} {'max|dtheta|':>12} {'rel':>8} {'nfev1':>7} {'nfev2':>6} {'resets':>6} {'Qmin':>8} {'t[s]':>5}")
for k in [1.1e5, 1.1e7, 1.1e9]:
    for per_decade in [100/12, 100]:
        for (rtol, atol) in [(1e-8, 1e-10), (1e-11, 1e-13), (5e-14, 1e-16)]:
            r = run(k, 30.0, per_decade, rtol, atol)
            print(f"{r['span']:8.2g} {per_decade:8.3g} {rtol:7.0e} {atol:7.0e} {'stg2':>7} {r['maxerr']:12.3e} {r['maxerr']/r['span']:8.1e} {r['n1']:7d} {r['n2']:6d} {r['resets']:6d} {r['Qmin']:8.3g} {r['dt']:5.1f}")
    if k < 1e8:
        for (rtol, atol) in [(1e-8, 1e-10), (5e-14, 1e-16)]:
            r = run(k, 30.0, 100/12, rtol, atol, force_stage1=True)
            print(f"{r['span']:8.2g} {100/12:8.3g} {rtol:7.0e} {atol:7.0e} {'stg1':>7} {r['maxerr']:12.3e} {r['maxerr']/r['span']:8.1e} {r['n1']:7d} {r['n2']:6d} {r['resets']:6d} {'-':>8} {r['dt']:5.1f}")
