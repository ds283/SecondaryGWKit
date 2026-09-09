"""Actual production phase path (integrate_phase_function) on the real LambdaCDM background, k=1e5/Mpc,
from 3 e-folds sub-horizon to z=0.1 on the response grid (100/12 per decade), vs mpmath reference."""
import sys, time, numpy as np, mpmath as mp
sys.path.insert(0, __file__.rsplit('/',1)[0])
from realbg import *
from common import grid, zarr, key
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq as om2, Gk_d_ln_omegaEff_dz as dlom
import Quadrature.integrators.WKB_phase_function as W
from scipy.integrate import quad
k = float(sys.argv[1]) if len(sys.argv) > 1 else 1e5
rtol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
from scipy.optimize import brentq
xof = lambda z: k*(1+z)/cosmo.Hubble(z)
z_exit = brentq(lambda z: np.log(xof(z)), 1.0, 1e18); z_e3 = brentq(lambda z: np.log(xof(z))-3.0, 1.0, z_exit)
zi = z_e3
zs = grid((1+zi)/10**(12/100)-1, 0.1, 100/12)
meta = {}
t0 = time.perf_counter()
out = W.integrate_phase_function(model, key(k), zi, zarr(zs), om2, dlom, om2(model, k, zi), atol, rtol, meta, "t4b", "G")
dt = time.perf_counter()-t0
n1 = out["stage_1_data"].RHS_evaluations if out["stage_1_data"] else 0
n2 = out["stage_2_data"].RHS_evaluations if out["stage_2_data"] else 0
# reference: theta(z) = -(k*[tau(z)-tau(zi)] + residual), sign: dtheta/dz=+omega so theta decreases toward low z
def resid(z):
    f = lambda u: (lambda zz, kH, o2: (o2-kH*kH)/(np.sqrt(o2)+kH)*np.exp(u))(np.exp(u)-1, k/cosmo.Hubble(np.exp(u)-1), om2(model, k, np.exp(u)-1))
    r, _ = quad(f, np.log(1+z), np.log(1+zi), limit=500, epsrel=1e-10, epsabs=1e-14); return r
idx = list(range(0, len(zs), max(1, len(zs)//12))) + [len(zs)-1]
print(f"k={k:.3g}/Mpc rtol={rtol:.0e} atol={atol:.0e}: z_e3={zi:.4g}, {len(zs)} response samples, stage1 nfev={n1}, stage2 nfev={n2}, resets={meta.get('phase_cycle_events')}, stage1->2 at z={meta.get('stage1_z_terminate')}, Qmin={meta.get('stage_2_smallest_Q')}, time {dt:.1f}s")
print(f"{'z':>10} {'theta_ref':>16} {'err[rad]':>10} {'rel':>8}")
worst = 0
for i in idx:
    z = zs[i]
    ref = -(mp.mpf(k)*tau_increment_mp(z, zi) + mp.mpf(resid(z)))
    got = mp.mpf(int(out["theta_div_2pi_sample"][i]))*2*mp.pi + mp.mpf(float(out["theta_mod_2pi_sample"][i]))
    e = float(got - ref); worst = max(worst, abs(e))
    print(f"{z:10.4g} {float(ref):16.8g} {e:10.3e} {abs(e)/abs(float(ref)):8.1e}")
print(f"max |err| over checked samples: {worst:.3e} rad")
