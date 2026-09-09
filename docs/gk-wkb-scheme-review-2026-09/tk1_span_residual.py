"""T_k: production phase spans, residual over the sound-horizon leading term, friction integral, on LambdaCDM and QCD."""
import sys, numpy as np
from math import log, exp, sqrt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import make_interp_spline
sys.path.insert(0, __file__.rsplit('/',1)[0])
from realbg import cosmo, model as lcdm
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz
from ComputeTargets.BackgroundModel import ModelFunctions
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from Units import Mpc_units
# --- d ln omega/dz check against finite differences (LambdaCDM, k=1e6)
k = 1e6
for z in [1e9, 1e6, 1e3, 10.0, 0.5]:
    h = 1e-4*(1+z); f = lambda zz: 0.5*log(Tk_omegaEff_sq(lcdm, k, zz))
    fd = (f(z+h) - f(z-h))/(2*h); an = Tk_d_ln_omegaEff_dz(lcdm, k, z)
    print(f"d ln omega_T/dz at z={z:.3g}: closed form {an:.10e}, finite difference {fd:.10e}, rel diff {abs(an-fd)/abs(an):.1e}")
# --- QCD model with spline-derived eps, w derivatives (as BackgroundModel does)
qcd = QCD_Cosmology(store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20)
u = np.linspace(log(1.05), log(1e16), 15*2000)
lnH = make_interp_spline(u, [log(qcd.Hubble(exp(v)-1)) for v in u], k=5); wS = make_interp_spline(u, [qcd.wPerturbations(exp(v)-1) for v in u], k=5)
d1, d2, d3 = lnH.derivative(1), lnH.derivative(2), lnH.derivative(3); w1, w2 = wS.derivative(1), wS.derivative(2)
L = lambda z: log(1+z)
def dz(fu, z, n):  # convert d^n/du^n to d^n/dz^n for n=1,2 (u=ln(1+z))
    s = 1+z
    if n == 1: return float(fu(L(z)))/s
    return (float(d2(L(z))) - float(d1(L(z))))/s**2
class M: pass
qm = M(); qm.functions = ModelFunctions(
    Hubble=qcd.Hubble, epsilon=lambda z: float(d1(L(z))),
    d_epsilon_dz=lambda z: float(d2(L(z)))/(1+z),
    d2_epsilon_dz2=lambda z: (float(d3(L(z))) - float(d2(L(z))))/(1+z)**2,
    wBackground=qcd.wBackground, wPerturbations=qcd.wPerturbations, tau=None, T_photon=qcd.T_photon,
    d_lnH_dz=lambda z: float(d1(L(z)))/(1+z), d2_lnH_dz2=None, d3_lnH_dz3=None,
    d_wPerturbations_dz=lambda z: float(w1(L(z)))/(1+z),
    d2_wPerturbations_dz2=lambda z: (float(w2(L(z))) - float(w1(L(z))))/(1+z)**2)
ZEND = 0.1
for label, m, c in [("LambdaCDM", lcdm, cosmo), ("QCD", qm, qcd)]:
    H = c.Hubble; w = c.wPerturbations
    print(f"--- {label}: x_T = k c_s/(aH); spans from the 3-e-fold point (x=e^3) to z=0.1")
    print(f"{'k':>6} {'z_e3':>9} {'theta_T span':>13} {'kc_s tau_s(0.1)':>15} {'residual rho_T':>14} {'F(0.1)':>8} {'eps*theta':>9} {'1e-8*theta':>10}")
    for k in [1e5, 1e6, 1e7, 1e8, 3e8]:
        xof = lambda z: k*(1+z)/H(z)
        z_exit = brentq(lambda z: log(xof(z)), 1.0, 1e19); z_e3 = brentq(lambda z: log(xof(z))-3, 1.0, z_exit)
        lead = quad(lambda v: (lambda z: k*sqrt(w(z))/H(z)*exp(v))(exp(v)-1), log(1+ZEND), log(1+z_e3), limit=800, epsrel=1e-11)[0]
        def fres(v):
            z = exp(v)-1; kcH = k*sqrt(w(z))/H(z); o2 = Tk_omegaEff_sq(m, k, z)
            return (o2 - kcH*kcH)/(sqrt(o2)+kcH)*exp(v)
        res = quad(fres, log(1+ZEND), log(1+z_e3), limit=800, epsrel=1e-9)[0]
        F = quad(lambda v: 1.5*(1+w(exp(v)-1)), log(1+ZEND), log(1+z_e3), limit=800)[0]
        xT_end = k*sqrt(w(ZEND))/H(ZEND)*(1+ZEND)
        print(f"{k:6.0e} {z_e3:9.3g} {lead+res:13.6g} {lead:15.6g} {res:14.4g} {F:8.3f} {2.2e-16*lead:9.1e} {1e-8*lead:10.1e}   [x_T(z=0.1)={xT_end:.3g}]")
