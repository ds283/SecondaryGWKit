"""QCD_Cosmology production model: size of the C-residual phase and of the LG next-order phase correction."""
import numpy as np, time
from math import log, exp
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq
from scipy.integrate import quad
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from Units import Mpc_units
t0 = time.perf_counter()
cosmo = QCD_Cosmology(store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20)
print(f"built QCD_Cosmology in {time.perf_counter()-t0:.1f}s; has d_lnH_dz? {hasattr(cosmo,'d_lnH_dz')}")
H = cosmo.Hubble
# eps = d lnH / d ln(1+z) from a fine grid (2000/decade), quintic spline
u = np.linspace(log(1.1), log(1e15), int(14*2000)); lnH = np.array([log(H(exp(v)-1)) for v in u])
eps_s = make_interp_spline(u, lnH, k=5).derivative(); epsp_s = eps_s.derivative()  # d eps/du
eps = lambda z: float(eps_s(log(1+z))); deps_dz = lambda z: float(epsp_s(log(1+z)))/(1+z)
zz = np.exp(u[500:-500]) - 1; ev = np.array([eps(z) for z in zz])
i = np.argmax(np.abs(ev-2.0)*(zz>1e8)); print(f"max |eps-2| for z>1e8: {abs(ev[i]-2):.3f} at z={zz[i]:.3g} (T~{cosmo.T_photon(zz[i])*1e-6/ (Mpc_units().Kelvin*1e-6) if hasattr(Mpc_units(),'Kelvin') else 0:.3g} K-units)")
def C(z):
    e = eps(z); s = 1+z; return -deps_dz(z)/2/s + (1.5*e - e*e/4 - 2)/(s*s)
def Rfun(z):
    e = eps(z); s = 1+z; return (e*e/4 - e/2)/(s*s) + deps_dz(z)/(2*s)
for k in [1e7, 1e8, 3e8]:
    xof = lambda z: k*(1+z)/H(z)
    z_exit = brentq(lambda z: log(xof(z)), 1.0, 1e19); z_e3 = brentq(lambda z: log(xof(z))-3, 1.0, z_exit)
    z_qcd = 6e11
    res = quad(lambda v: (lambda z: C(z)/(2*k/H(z))*exp(v))(exp(v)-1), log(1.1), log(1+z_e3), limit=800, epsrel=1e-8)[0]
    lg = quad(lambda v: (lambda z: Rfun(z)/(2*k/H(z))*exp(v))(exp(v)-1), log(1.1), log(1+z_e3), limit=800, epsrel=1e-8)[0]
    print(f"k={k:.0e}: z_e3={z_e3:.3g} (x at z=6e11: {xof(z_qcd):.3g}); C-residual phase over WKB region = {res:.3g} rad; LG next-order phase correction int R/(2 omega) dz = {lg:.3g} rad; C at z=6e11: {C(z_qcd)*(1+z_qcd)**2:.3g}/(1+z)^2")
