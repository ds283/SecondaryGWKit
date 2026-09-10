"""Per-object independent phase solves: the sample-to-sample noise seen by the consumer's theta(z_source) spline."""
import sys, numpy as np, mpmath as mp
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq as om2, Gk_d_ln_omegaEff_dz as dlom
import Quadrature.integrators.WKB_phase_function as W
from LiouvilleGreen.phase_spline import phase_spline
from LiouvilleGreen.WKBtools import WKB_mod_2pi
k = 1.1e7; z_r = 0.1
for x_lo, x_hi in [(1e3, 3e4)]:
    ss = np.geomspace(k/x_hi, k/x_lo, int(np.log10(x_hi/x_lo)*100)+1)  # ascending s = 1+z_source
    got, ex = [], []
    for s in ss:
        zi = s - 1.0; meta = {}
        out = W.integrate_phase_function(RadModel(), key(k), zi, zarr([z_r]), om2, dlom, om2(RadModel(), k, zi), 1e-10, 1e-8, meta, "t7", "G")
        got.append(mp.mpf(int(out["theta_div_2pi_sample"][0]))*2*mp.pi + mp.mpf(float(out["theta_mod_2pi_sample"][0])))
        ex.append(exact_theta(k, zi, z_r))
    err = np.array([float(g - e) for g, e in zip(got, ex)])
    inc = np.array([float(ex[i+1]-ex[i]) for i in range(len(ex)-1)])
    jit = np.diff(err)
    print(f"k={k:.2g}, x_source in [{x_lo:.0e},{x_hi:.0e}], {len(ss)} independent solves at production tolerances:")
    print(f"  max|err| = {np.max(np.abs(err)):.2e} rad; neighbour-to-neighbour error jitter max {np.max(np.abs(jit)):.2e} rad vs true per-sample increment {np.min(np.abs(inc)):.3g}..{np.max(np.abs(inc)):.3g} rad -> relative roughness {np.max(np.abs(jit)/np.abs(inc)):.1e}")
    # consumer-style spline of theta(log(1+z_s)) through solved vs exact samples; compare theta' (log derivative)
    us = np.log(ss)
    def spl(vals):
        pairs = [WKB_mod_2pi(float(v)) for v in vals]; d, m = map(list, zip(*pairs)); b = d[0]; d = [v-b for v in d]
        return phase_spline(list(us), d, m, x_is_log=True, x_is_redshift=True, increasing=False, chunk_step=None, chunk_logstep=None)
    sg, se = spl(got), spl(ex)
    mids = 0.5*(us[:-1]+us[1:])[5:-5]
    dg = np.array([sg.theta_deriv(float(u), x_is_log=True, log_derivative=True) for u in mids])
    de = np.array([se.theta_deriv(float(u), x_is_log=True, log_derivative=True) for u in mids])
    dtrue = np.array([-k/np.exp(u) for u in mids])  # d theta/d log s = -k/s
    print(f"  spline theta' (log-derivative) relative error at interior midpoints: from exact samples {np.max(np.abs(de/dtrue-1)):.1e}, from solved samples {np.max(np.abs(dg/dtrue-1)):.1e}")
    vg = np.array([sg.raw_theta(float(u), x_is_log=True) for u in mids]); ve = np.array([se.raw_theta(float(u), x_is_log=True) for u in mids])
    print(f"  spline theta value: max |spline(solved) - spline(exact)| at midpoints = {np.max(np.abs(vg-ve)):.2e} rad")
