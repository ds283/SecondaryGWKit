"""Production phase spans on the real Planck2018 LambdaCDM background (no Ray)."""
import numpy as np
from math import log, exp, sqrt
from scipy.integrate import quad
from scipy.optimize import brentq
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units
from ComputeTargets.BackgroundModel import ModelFunctions
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq

cosmo = LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
class M: pass
m = M()
m.functions = ModelFunctions(
    Hubble=cosmo.Hubble,
    epsilon=lambda z: (1.0 + z) * cosmo.d_lnH_dz(z),
    d_epsilon_dz=lambda z: cosmo.d_lnH_dz(z) + (1.0 + z) * cosmo.d2_lnH_dz2(z),
    d2_epsilon_dz2=lambda z: 2.0 * cosmo.d2_lnH_dz2(z) + (1.0 + z) * cosmo.d3_lnH_dz3(z),
    wBackground=cosmo.wBackground, wPerturbations=cosmo.wPerturbations, tau=None,
    T_photon=cosmo.T_photon, d_lnH_dz=cosmo.d_lnH_dz, d2_lnH_dz2=cosmo.d2_lnH_dz2,
    d3_lnH_dz3=cosmo.d3_lnH_dz3, d_wPerturbations_dz=cosmo.d_wPerturbations_dz,
    d2_wPerturbations_dz2=cosmo.d2_wPerturbations_dz2)
H = cosmo.Hubble
ZEND = 0.1
def x_of(k, z):  # k/(aH) with a0 absorbed: k(1+z)/H
    return k * (1 + z) / H(z)
def C_of(z):
    eps = m.functions.epsilon(z); epsP = m.functions.d_epsilon_dz(z); s = 1 + z
    return -epsP / 2.0 / s + (1.5 * eps - eps * eps / 4.0 - 2.0) / (s * s)
print(f"{'k[1/Mpc]':>9} {'z_exit':>9} {'z_e3':>9} {'x(z=0.1)':>10} {'theta_lead(e3->0.1)':>19} {'residual':>10} {'eps*theta':>10} {'1e-8*theta':>10}")
for k in [1e5, 1e6, 1e7, 1e8, 3e8]:
    z_exit = brentq(lambda z: log(x_of(k, z)), 1.0, 1e18)
    z_e3 = brentq(lambda z: log(x_of(k, z)) - 3.0, 1.0, z_exit)
    # leading phase: int k/H dz  in u=log(1+z)
    f_lead = lambda u: k / H(exp(u) - 1) * exp(u)
    lead, _ = quad(f_lead, log(1 + ZEND), log(1 + z_e3), limit=500, epsrel=1e-12)
    def f_res(u):
        z = exp(u) - 1; kH = k / H(z); om2 = Gk_omegaEff_sq(m, k, z)
        return (om2 - kH * kH) / (sqrt(om2) + kH) * exp(u)
    res, _ = quad(f_res, log(1 + ZEND), log(1 + z_e3), limit=500, epsrel=1e-10)
    print(f"{k:9.3g} {z_exit:9.3g} {z_e3:9.3g} {x_of(k, ZEND):10.3g} {lead:19.6g} {res:10.4g} {2.2e-16*lead:10.2g} {1e-8*lead:10.2g}")
# residual at handover scale: from z_e3 to z where x=1e3
k = 1e6
z_exit = brentq(lambda z: log(x_of(k, z)), 1.0, 1e18)
for xi in [20, 30, 100, 1000]:
    z_a = brentq(lambda z: log(x_of(k, z)) - log(xi), 1.0, z_exit)
    def f_res(u):
        z = exp(u) - 1; kH = k / H(z); om2 = Gk_omegaEff_sq(m, k, z)
        return (om2 - kH * kH) / (sqrt(om2) + kH) * exp(u)
    res, _ = quad(f_res, log(1 + ZEND), log(1 + z_a), limit=500, epsrel=1e-10)
    print(f"k=1e6: residual phase from x={xi} to z=0.1: {res:.4g} rad ; C at that z = {C_of(z_a):.3g}, 1/x={1/xi:.3g}")
