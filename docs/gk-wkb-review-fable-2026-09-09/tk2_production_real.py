"""T_k production phase+friction path (WKB_phase_function with Tk frequency, friction_RHS) on LambdaCDM, vs mpmath."""
import sys, time, numpy as np, mpmath as mp
from math import log, exp, sqrt
from types import SimpleNamespace as NS
sys.path.insert(0, __file__.rsplit('/',1)[0])
from realbg import cosmo, model, H_mp
from common import grid, zarr
from CosmologyConcepts import redshift, redshift_array
from scipy.optimize import brentq
from scipy.integrate import quad
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz
import ComputeTargets.TkWKBIntegration; TK = sys.modules["ComputeTargets.TkWKBIntegration"]
import Quadrature.integrators.WKB_phase_function as W
W.check_units = lambda *a, **kw: None
run = W.WKB_phase_function._function
Om, Or = mp.mpf(cosmo.omega_m), mp.mpf(cosmo.omega_r)
def w_mp(z): s = 1+mp.mpf(z); return (Or*s/3)/(Om + Or*s)
def lead_mp(z_lo, z_hi, k):
    f = lambda u: mp.mpf(k)*mp.sqrt(w_mp(mp.exp(u)-1))/H_mp(mp.exp(u)-1)*mp.exp(u)
    return mp.quad(f, [mp.log(1+mp.mpf(z_lo)), mp.log(1+mp.mpf(z_hi))])
def F_mp(z_lo, z_hi):
    f = lambda u: mp.mpf(1.5)*(1+w_mp(mp.exp(u)-1))
    return mp.quad(f, [mp.log(1+mp.mpf(z_lo)), mp.log(1+mp.mpf(z_hi))])
k = float(sys.argv[1]) if len(sys.argv) > 1 else 1e5
H = cosmo.Hubble; xof = lambda z: k*(1+z)/H(z)
z_exit = brentq(lambda z: log(xof(z)), 1.0, 1e19); zi = brentq(lambda z: log(xof(z))-3, 1.0, z_exit)
zs = grid((1+zi)/10**(1/100)-1, 0.1, 100)
proxy = NS(get=lambda: model); kmock = NS(k=NS(k=k, k_inv_Mpc=k, store_id=0))
t0 = time.perf_counter()
out = run(proxy, kmock, zi, zarr(zs), omega_sq=Tk_omegaEff_sq, d_ln_omega_dz=Tk_d_ln_omegaEff_dz, friction=TK.friction_RHS,
          atol=1e-10, rtol=1e-8, task_label="tk2", object_label="T")
dt = time.perf_counter()-t0
n1 = out["stage_1_data"].RHS_evaluations if out["stage_1_data"] else 0; n2 = out["stage_2_data"].RHS_evaluations if out["stage_2_data"] else 0
nf = out["friction_data"].RHS_evaluations; meta = out["metadata"]
print(f"k={k:.0e}/Mpc: z_init=z_e3={zi:.4g}, {len(zs)} source samples to z=0.1; stage1 nfev={n1} ({meta.get('phase_cycle_events')} resets), stage2 nfev={n2}, friction nfev={nf}; Qmin={meta.get('stage_2_smallest_Q')}; time {dt:.1f}s")
def resid(z):
    def f(v):
        zz = exp(v)-1; kcH = k*sqrt(cosmo.wPerturbations(zz))/H(zz); o2 = Tk_omegaEff_sq(model, k, zz)
        return (o2 - kcH*kcH)/(sqrt(o2)+kcH)*exp(v)
    return quad(f, log(1+z), log(1+zi), limit=800, epsrel=1e-10, epsabs=1e-14)[0]
idx = list(range(0, len(zs), max(1, len(zs)//12))) + [len(zs)-1]
print(f"{'z':>10} {'x_T':>9} {'theta_ref':>16} {'phase err':>10} {'rel':>8} {'friction F':>11} {'dF':>9}")
worst = 0; worstF = 0
for i in idx:
    z = zs[i]
    ref = -(mp.mpf(k)*0 + lead_mp(z, zi, k) + mp.mpf(resid(z)))
    got = mp.mpf(int(out["theta_div_2pi_sample"][i]))*2*mp.pi + mp.mpf(float(out["theta_mod_2pi_sample"][i]))
    e = float(got - ref); worst = max(worst, abs(e))
    Fref = -F_mp(z, zi); dF = float(mp.mpf(float(out["friction_sample"][i])) - Fref); worstF = max(worstF, abs(dF))
    print(f"{z:10.4g} {k*sqrt(cosmo.wPerturbations(z))*(1+z)/H(z):9.3g} {float(ref):16.8g} {e:10.3e} {abs(e)/abs(float(ref)):8.1e} {float(Fref):11.5f} {dF:9.2e}")
print(f"max |phase err| = {worst:.3e} rad; max |dF| (= relative amplitude error from friction) = {worstF:.2e}")
